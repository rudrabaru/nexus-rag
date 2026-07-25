import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

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
            loop = asyncio.get_running_loop()
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
            app.state.query_semaphore = asyncio.Semaphore(10)

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
                    pass # Metric check omitted for Qdrant (managed internally)

                qdrant_count = vector_store.get_collection_size() if vector_store else 0

                registry_docs = registry.list_documents(tenant_id=None)
                registry_chunk_count = sum(
                    len(d.get("chunk_ids", [])) for d in registry_docs
                )

                logger.info(
                    f"Startup check: Qdrant chunks={qdrant_count}, Registry chunks={registry_chunk_count}"
                )
                if qdrant_count != registry_chunk_count:
                    logger.warning(
                        f"CRITICAL STATE DIVERGENCE: Qdrant has {qdrant_count} chunks but Registry has {registry_chunk_count} chunks. "
                        f"In a stateless environment like Render, this is expected if the ephemeral SQLite DB is wiped but Qdrant persists."
                    )
                    
                    if registry_chunk_count == 0 and qdrant_count > 0:
                        logger.info("Auto-rebuilding local SQLite registry from Qdrant...")
                        try:
                            rebuilt_count = await loop.run_in_executor(None, registry.rebuild_registry_from_qdrant, vector_store)
                            logger.info(f"Successfully auto-rebuilt {rebuilt_count} chunks into local SQLite registry.")
                        except Exception as e:
                            logger.error(f"Auto-rebuild failed: {e}")
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
