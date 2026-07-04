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

logger = logging.getLogger(__name__)

# Global state
app_state = {}

from dataclasses import dataclass


@dataclass
class PipelineComponents:
    retriever: object
    reranker: object
    generator: object
    evaluator: object
    provider: str
    model_name: str
    rewriter: object


def _init_components() -> PipelineComponents:
    """Shared factory function to initialize core pipeline components."""
    distance_metric = "cosine"

    collection_name = os.environ.get("CHROMA_COLLECTION", "unified_corpus")
    db_manager = ChromaDBManager(
        collection_name=collection_name, distance_metric=distance_metric
    )
    retriever = DenseRetriever(vector_store=db_manager)

    bm25_path = Path("data/bm25_index.pkl")
    if bm25_path.exists():
        from src.retrieving.retriever import HybridRetriever

        retriever = HybridRetriever(
            dense_retriever=retriever, bm25_index_path=str(bm25_path)
        )
        logger.info("HybridRetriever loaded with BM25 + Dense.")

    reranker = OptionalReranker()

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    model_name = os.environ.get(
        "LLM_MODEL_NAME",
        "gemini-3.1-flash-lite" if provider == "gemini" else "llama-3.3-70b-versatile",
    )

    fallback_config = None
    if provider == "gemini" and os.environ.get("GROQ_API_KEY"):
        logger.info(
            "GROQ_API_KEY detected. Configuring Groq as automatic fallback for rate limits."
        )
        fallback_config = {
            "provider": "groq",
            "model_name": "llama-3.3-70b-versatile",
            "max_output_tokens": 4096,
            "temperature": 0.1,
        }

    config = GenerationConfig(
        provider=provider, model_name=model_name, fallback_config=fallback_config
    )
    generator = RAGGenerator(retriever=retriever, config=config)
    evaluator = FaithfulnessEvaluator(config=config)
    rewriter = QueryRewriter(config=config)

    return PipelineComponents(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        evaluator=evaluator,
        provider=provider,
        model_name=model_name,
        rewriter=rewriter,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run model loading in a background thread so the server
    # binds to the port immediately and /health starts passing.
    import asyncio

    async def _load_models_async():
        try:
            loop = asyncio.get_event_loop()
            components = await loop.run_in_executor(None, _init_components)

            app_state["retriever"] = components.retriever
            app_state["reranker"] = components.reranker
            app_state["generator"] = components.generator
            app_state["evaluator"] = components.evaluator
            app_state["rewriter"] = components.rewriter
            registry = DocumentRegistry()
            app_state["registry"] = registry
            app_state["ingestion_semaphore"] = asyncio.Semaphore(3)

            # Reset stuck jobs on startup (Fix for C3 race condition)
            try:
                registry.reset_stuck_jobs()
            except Exception as e:
                logger.error(f"Failed to reset stuck jobs: {e}")

            try:
                vector_store = getattr(retriever, "vector_store", None)
                if not vector_store and hasattr(retriever, "dense_retriever"):
                    vector_store = getattr(
                        retriever.dense_retriever, "vector_store", None
                    )
                chroma_count = vector_store.collection.count() if vector_store else 0

                registry_docs = registry.list_documents()
                registry_chunk_count = sum(
                    len(d.get("chunk_ids", [])) for d in registry_docs
                )

                logger.info(
                    f"Startup check: ChromaDB chunks={chroma_count}, Registry chunks={registry_chunk_count}"
                )
                if chroma_count != registry_chunk_count:
                    logger.warning(
                        f"CRITICAL STATE DIVERGENCE: ChromaDB has {chroma_count} chunks but Registry has {registry_chunk_count} chunks."
                    )

                if hasattr(retriever, "bm25") and retriever.bm25:
                    bm25_count = len(retriever.bm25.corpus)
                    logger.info(f"Startup check: BM25 corpus size={bm25_count}")

                if Path("data/bm25_dirty.flag").exists():
                    logger.info(
                        "BM25 index is marked dirty at startup. Rebuilding in background..."
                    )
                    from src.ingestion.pipeline import IncrementalIngestionPipeline

                    loop.run_in_executor(
                        None, IncrementalIngestionPipeline()._rebuild_bm25
                    )
            except Exception as e:
                logger.error(f"Failed to perform startup consistency check: {e}")

            logger.info(
                f"RAG Pipeline API ready. Provider: {provider}, Model: {model_name}"
            )
            app_state["ready"] = True
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            app_state["init_error"] = str(e)

    # Fire and forget the background loading
    task = asyncio.create_task(_load_models_async())

    yield
    # Shutdown
    app_state.clear()
