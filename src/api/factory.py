import os
import logging
from dataclasses import dataclass
from typing import Optional, Union

from src.retrieving.vector_store import QdrantManager
from src.retrieving.retriever import DenseRetriever, OptionalReranker, HybridRetriever
from src.generating.models import GenerationConfig
from src.generating.generator import RAGGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter
from src.embedding.generator import EmbeddingGenerator
from src.embedding.config import EmbeddingConfig

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

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
    registry = DocumentRegistry()

    retriever = HybridRetriever(
        dense_retriever=retriever, registry=registry
    )
    logger.info("HybridRetriever loaded with SQLite FTS5 + Dense.")

    enable_reranker = os.environ.get("ENABLE_RERANKER", "true").lower() == "true"
    if enable_reranker:
        reranker = OptionalReranker()
    else:
        reranker = None
        logger.info("Reranker disabled via ENABLE_RERANKER env var.")

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    model_name = os.environ.get("LLM_MODEL_NAME")
    if not model_name:
        model_name = (
            DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_GROQ_MODEL
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
    generator = RAGGenerator(config=config)
    evaluator = FaithfulnessEvaluator(config=config)
    
    rewriter_config = GenerationConfig(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
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
