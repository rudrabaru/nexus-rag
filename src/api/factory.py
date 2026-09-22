import logging
from dataclasses import dataclass
from typing import Optional, Union

from src.config import get_settings
from src.retrieving.vector_store import QdrantManager
from src.retrieving.retriever import DenseRetriever, OptionalReranker, HybridRetriever
from src.generating.models import GenerationConfig
from src.generating.generator import RAGGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter
from src.embedding.generator import EmbeddingGenerator
from src.embedding.config import EmbeddingConfig

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
# llama-3.1-8b-instant was retired from Groq (404 NotFoundError, confirmed live
# 2026-09-22 against the project's key; the key's /models list no longer carries it).
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

logger = logging.getLogger(__name__)

@dataclass
class PipelineComponents:
    retriever: Union[DenseRetriever, HybridRetriever]
    reranker: Optional[OptionalReranker]
    generator: RAGGenerator
    evaluator: FaithfulnessEvaluator
    provider: str
    model_name: str
    rewriter: QueryRewriter
    embedding_generator: EmbeddingGenerator


def _init_components() -> PipelineComponents:
    """Shared factory function to initialize core pipeline components."""
    distance_metric = "cosine"
    
    try:
        db_manager = QdrantManager(
            distance_metric=distance_metric
        )
        logger.info("QdrantManager initialized successfully as primary vector store.")
    except Exception as e:
        logger.critical(f"Qdrant failed to initialize: {e}")
        raise e
        
    retriever = DenseRetriever(vector_store=db_manager)

    from src.registry.database import DocumentRegistry
    
    # Note: We create a distinct DocumentRegistry instance here purely to furnish the
    # HybridRetriever with an FTS search handle. The primary application registry is 
    # instantiated in startup.py and lives on app.state.registry. Both instances use
    # the same underlying SQLite WAL file so this is completely safe.
    registry = DocumentRegistry()

    retriever = HybridRetriever(
        dense_retriever=retriever, registry=registry
    )
    logger.info("HybridRetriever loaded with SQLite FTS5 + Dense.")

    settings = get_settings()

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
    
    embed_config = EmbeddingConfig()
    embedding_generator = EmbeddingGenerator(embed_config)

    return PipelineComponents(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        evaluator=evaluator,
        provider=provider,
        model_name=model_name,
        rewriter=rewriter,
        embedding_generator=embedding_generator,
    )
