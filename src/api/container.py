import logging
from dataclasses import dataclass

from src.config import get_settings
from src.db.engine import get_async_engine, get_sync_engine
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.generator import RAGGenerator
from src.generating.models import GenerationConfig
from src.generating.query_rewriter import QueryRewriter
from src.llm.roles import role_fields
from src.retrieving.pipeline import RetrievalResources

logger = logging.getLogger(__name__)

REWRITE_MAX_OUTPUT_TOKENS = 1024  # a rewritten query is a sentence; the budget only has to cover a reasoning model's thinking


@dataclass
class PipelineComponents:
    retrieval: RetrievalResources
    generator: RAGGenerator
    evaluator: FaithfulnessEvaluator
    rewriter: QueryRewriter

    @property
    def model(self) -> str:
        return self.generator.config.model_string


def build_components() -> PipelineComponents:
    """Shared factory function to initialize core pipeline components."""
    settings = get_settings()

    retrieval = RetrievalResources(settings, get_sync_engine(), get_async_engine())
    reranker = settings.effective_reranker
    if reranker == "flashrank":
        retrieval.reranker(reranker).load()  # model load (and first-run download) at startup, not on a query
    logger.info(
        f"Retrieval: index {retrieval.default_index_id}, default strategy {settings.retrieval_strategy}, "
        f"reranker {reranker or 'off'}."
    )

    chat = GenerationConfig(**role_fields(settings, "chat"))
    if chat.fallback_config:
        logger.info(f"Chat falls back to {chat.fallback_config['provider']}/{chat.fallback_config['model_name']} when {chat.model_string} fails.")

    return PipelineComponents(
        retrieval=retrieval,
        generator=RAGGenerator(config=chat),
        evaluator=FaithfulnessEvaluator(config=GenerationConfig(**role_fields(settings, "judge"))),
        rewriter=QueryRewriter(config=GenerationConfig(**role_fields(settings, "rewrite"), max_output_tokens=REWRITE_MAX_OUTPUT_TOKENS)),
    )
