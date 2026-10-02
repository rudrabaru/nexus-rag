import logging
from dataclasses import dataclass

from src.config import get_settings
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.generator import RAGGenerator
from src.generating.models import DEFAULT_GROQ_MODEL, GenerationConfig, default_model_name
from src.generating.query_rewriter import QueryRewriter
from src.registry.database import DocumentRegistry
from src.registry.engine import get_async_engine, get_sync_engine
from src.retrieving.pipeline import RetrievalResources


logger = logging.getLogger(__name__)


@dataclass
class PipelineComponents:
    retrieval: RetrievalResources
    registry: DocumentRegistry
    generator: RAGGenerator
    evaluator: FaithfulnessEvaluator
    provider: str
    model_name: str
    rewriter: QueryRewriter


def _init_components() -> PipelineComponents:
    """Shared factory function to initialize core pipeline components."""
    settings = get_settings()

    retrieval = RetrievalResources(settings, get_sync_engine(), get_async_engine())
    registry = DocumentRegistry(get_sync_engine())
    reranker = settings.effective_reranker
    if reranker == "flashrank":
        retrieval.reranker(reranker).load()  # model load (and first-run download) at startup, not on a query
    logger.info(
        f"Retrieval: index {retrieval.default_index_id}, default strategy {settings.retrieval_strategy}, "
        f"reranker {reranker or 'off'}."
    )

    provider = settings.llm_provider
    model_name = settings.llm_model_name or default_model_name(provider)

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

    return PipelineComponents(
        retrieval=retrieval,
        registry=registry,
        generator=generator,
        evaluator=evaluator,
        provider=provider,
        model_name=model_name,
        rewriter=rewriter,
    )
