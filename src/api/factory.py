import logging
from dataclasses import dataclass
from typing import Optional

from src.config import get_settings
from src.embedding.config import EmbeddingConfig
from src.embedding.generator import EmbeddingGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.generator import RAGGenerator
from src.generating.models import GenerationConfig
from src.generating.query_rewriter import QueryRewriter
from src.registry.database import DocumentRegistry
from src.registry.engine import get_async_engine, get_sync_engine
from src.retrieving.chunk_store import ChunkStore
from src.retrieving.retriever import DenseRetriever, HybridRetriever, OptionalReranker, SparseRetriever

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
# llama-3.1-8b-instant was retired from Groq (404 NotFoundError, confirmed live
# 2026-09-22 against the project's key; the key's /models list no longer carries it).
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

logger = logging.getLogger(__name__)


@dataclass
class PipelineComponents:
    chunk_store: ChunkStore
    registry: DocumentRegistry
    retriever: HybridRetriever
    reranker: Optional[OptionalReranker]
    generator: RAGGenerator
    evaluator: FaithfulnessEvaluator
    provider: str
    model_name: str
    rewriter: QueryRewriter
    embedding_generator: EmbeddingGenerator


def _init_components() -> PipelineComponents:
    """Shared factory function to initialize core pipeline components."""
    settings = get_settings()

    chunk_store = ChunkStore(get_sync_engine(), get_async_engine())
    registry = DocumentRegistry(get_sync_engine())
    retriever = HybridRetriever(DenseRetriever(chunk_store), SparseRetriever(chunk_store))
    logger.info("HybridRetriever loaded: pgvector HNSW + Postgres full-text search.")

    if settings.enable_reranker:
        reranker = OptionalReranker()
    else:
        reranker = None
        logger.info("Reranker disabled via ENABLE_RERANKER env var.")

    provider = settings.llm_provider
    model_name = settings.llm_model_name or (
        DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_GROQ_MODEL
    )

    fallback_config = None
    if provider == "gemini" and settings.groq_api_key:
        logger.info(
            "GROQ_API_KEY detected. Configuring Groq as automatic fallback for rate limits."
        )
        fallback_config = {
            "provider": "groq",
            "model_name": DEFAULT_GROQ_MODEL,
            "max_output_tokens": 4096,
            "temperature": 0.1,
        }

    config = GenerationConfig(
        provider=provider, model_name=model_name, fallback_config=fallback_config
    )
    generator = RAGGenerator(config=config)
    evaluator = FaithfulnessEvaluator(config=config)

    rewriter_config = GenerationConfig(
        provider="groq",
        model_name=DEFAULT_GROQ_MODEL,
        temperature=0.1,
        fallback_config={
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "max_output_tokens": 1024,
            "temperature": 0.1,
        },
    )
    rewriter = QueryRewriter(config=rewriter_config)

    embedding_generator = EmbeddingGenerator(EmbeddingConfig())

    return PipelineComponents(
        chunk_store=chunk_store,
        registry=registry,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        evaluator=evaluator,
        provider=provider,
        model_name=model_name,
        rewriter=rewriter,
        embedding_generator=embedding_generator,
    )
