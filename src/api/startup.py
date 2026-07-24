import os
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.retrieving.vector_store import ChromaDBManager
from src.retrieving.retriever import DenseRetriever, OptionalReranker
from src.generating.models import GenerationConfig
from src.generating.generator import RAGGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter
from src.registry.database import DocumentRegistry
from src.api.factory import _init_components

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run model loading in a background thread so the server
    # binds to the port immediately and /health starts passing.
    import asyncio

    async def _load_models_async():
        try:
            loop = asyncio.get_event_loop()
            components = await loop.run_in_executor(None, _init_components)

            app.state.retriever = components.retriever
            app.state.reranker = components.reranker
            app.state.generator = components.generator
            app.state.evaluator = components.evaluator
            app.state.rewriter = components.rewriter
            app.state.embedding_generator = components.embedding_generator
            registry = DocumentRegistry()
            app.state.registry = registry
            
            from src.registry.auth_store import AuthStore
            from src.registry.metrics_store import MetricsStore
            from src.observability.logger import PipelineLogger
            app.state.auth_store = AuthStore(registry._get_conn)
            app.state.metrics_store = MetricsStore(registry._get_conn)
            app.state.pipeline_logger = PipelineLogger("nexus_rag", registry=registry)

            # Configurable via INGESTION_CONCURRENCY env var.
            # Production default: 3 (respects Jina 1000 RPM free tier).
            # Local dev: set INGESTION_CONCURRENCY=10 in .env to avoid artificial throttling.
            ingestion_concurrency = int(os.environ.get("INGESTION_CONCURRENCY", "3"))
            app.state.ingestion_semaphore = asyncio.Semaphore(ingestion_concurrency)
            app.state.embed_semaphore = asyncio.Semaphore(ingestion_concurrency)

            if hasattr(app.state, 'embedding_generator') and app.state.embedding_generator:
                app.state.embedding_generator.embed_semaphore = app.state.embed_semaphore

            # Reset stuck jobs on startup (Fix for C3 race condition)
            try:
                registry.reset_stuck_jobs()
            except Exception as e:
                logger.error(f"Failed to reset stuck jobs: {e}")

            try:
                vector_store = getattr(components.retriever, "vector_store", None)
                if not vector_store and hasattr(components.retriever, "dense_retriever"):
                    vector_store = getattr(
                        components.retriever.dense_retriever, "vector_store", None
                    )

                if vector_store:
                    actual_metric = vector_store.collection.metadata.get("hnsw:space", "unknown")
                    expected_metric = getattr(vector_store, "distance_metric", "cosine")
                    if actual_metric != "unknown" and actual_metric != expected_metric:
                        logger.critical(
                            f"CHROMADB DISTANCE METRIC MISMATCH: collection='{vector_store.collection.name}' "
                            f"has hnsw:space='{actual_metric}' but configured distance_metric='{expected_metric}'. "
                            f"All similarity scores will be incorrect. Delete and recreate the collection."
                        )

                chroma_count = vector_store.collection.count() if vector_store else 0

                registry_docs = registry.list_documents(tenant_id=None)
                registry_chunk_count = sum(
                    len(d.get("chunk_ids", [])) for d in registry_docs
                )

                logger.info(
                    f"Startup check: ChromaDB chunks={chroma_count}, Registry chunks={registry_chunk_count}"
                )
                if chroma_count != registry_chunk_count:
                    logger.warning(
                        f"CRITICAL STATE DIVERGENCE: ChromaDB has {chroma_count} chunks but Registry has {registry_chunk_count} chunks. Initiating auto-healing."
                    )
                    try:
                        chroma_ids = set()
                        if chroma_count > 0:
                            BATCH_SIZE = 5000
                            for offset in range(0, chroma_count, BATCH_SIZE):
                                batch_data = vector_store.collection.get(include=[], limit=BATCH_SIZE, offset=offset)
                                chroma_ids.update(batch_data.get("ids", []))
                        
                        deleted_docs = 0
                        for doc in registry_docs:
                            doc_chunk_ids = doc.get("chunk_ids", [])
                            if not doc_chunk_ids:
                                continue
                            
                            # If any chunk is missing in ChromaDB, the document didn't fully persist
                            missing_chunks = [cid for cid in doc_chunk_ids if cid not in chroma_ids]
                            if missing_chunks:
                                doc_id = doc.get("doc_id")
                                logger.info(f"Auto-healing: Deleting orphaned document {doc_id} ({len(missing_chunks)} missing chunks).")
                                registry.delete_document(doc_id)
                                deleted_docs += 1
                                
                        logger.info(f"Auto-healing complete. Deleted {deleted_docs} orphaned documents.")
                    except Exception as e:
                        logger.error(f"Failed to auto-heal state divergence: {e}")

            except Exception as e:
                logger.error(f"Failed to perform startup consistency check: {e}")

            logger.info(
                f"RAG Pipeline API ready. Provider: {components.provider}, Model: {components.model_name}"
            )
            app.state.ready = True
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            app.state.init_error = str(e)

    # Fire and forget the background loading
    task = asyncio.create_task(_load_models_async())
    app.state._startup_task = task

    yield
    # Shutdown
    if hasattr(app.state, "_startup_task"):
        app.state._startup_task.cancel()
        try:
            await app.state._startup_task
        except asyncio.CancelledError:
            pass
