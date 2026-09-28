import asyncio
import logging
import time
from typing import Any, Optional

from src.api.models.query_models import QueryRequest, SourceDocument
from src.config import get_settings
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.models import GenerationResult
from src.generating.query_rewriter import QueryRewriter
from src.retrieving.config import RetrievalConfig
from src.retrieving.models import RetrievalResult
from src.retrieving.pipeline import RetrievalResources, build_pipeline

logger = logging.getLogger(__name__)

# The reranker has always been handed four times the requested results to reorder.
CHAT_RERANK_POOL_FACTOR = 4


def chat_config(body: QueryRequest) -> RetrievalConfig:
    """Chat's configuration: the configured defaults, plus the request's top_k and reranker toggle."""
    settings = get_settings()
    reranker = settings.effective_reranker if body.use_reranker else None
    return RetrievalConfig(
        strategy=settings.retrieval_strategy.lower(),
        top_k=body.top_k,
        reranker=reranker,
        rerank_candidates=body.top_k * CHAT_RERANK_POOL_FACTOR,
    )


class QueryService:
    @staticmethod
    async def search_query(body: QueryRequest, rewriter: Optional[QueryRewriter]) -> str:
        """The text actually searched: optionally generalised, and made standalone when there is history."""
        query = body.query
        if rewriter and get_settings().enable_query_generalisation:
            query = await asyncio.to_thread(rewriter.generalise, query)
        if rewriter and body.history:
            query = await asyncio.to_thread(rewriter.rewrite, query, body.history)
        return query

    @staticmethod
    async def run_retrieval(
        body: QueryRequest,
        retrieval: RetrievalResources,
        rewriter: Optional[QueryRewriter],
        pipeline_logger: Any,
        tenant_id: str,
    ) -> RetrievalResult:
        config = chat_config(body)
        query = await QueryService.search_query(body, rewriter)
        start = time.time()
        result = await build_pipeline(config, retrieval).run(query, tenant_id, pipeline_logger=pipeline_logger)
        if pipeline_logger:
            pipeline_logger.log_event(
                "retrieval_complete", query_text=query, tenant_id=tenant_id, strategy=config.strategy,
                reranker=config.reranker, chunk_count=len(result.chunks), degraded=result.degraded,
                duration_ms=(time.time() - start) * 1000,
            )
        return result

    @staticmethod
    def construct_sources(result) -> list[SourceDocument]:
        return [
            SourceDocument(
                url=chunk.source_url or "",
                section=" > ".join(chunk.heading_path) if chunk.heading_path else "",
                similarity_score=chunk.similarity_score,
                chunk_preview=(
                    chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                ),
            )
            for chunk in result.context_window.included_chunks
        ]

    @staticmethod
    async def log_query(
        metrics_store, tenant_id: str, body: QueryRequest, retrieval_result, result: GenerationResult, latency_ms: float
    ) -> Optional[int]:
        """Persists the query's latency, tokens, serving provider and cost. Never fails the request."""
        if not metrics_store:
            return None
        details = {
            "top_k_requested": body.top_k,
            "faithfulness_reasoning": None,
            "retrieved_context": [
                {"chunk_id": c.chunk_id, "source_url": c.source_url, "similarity_score": c.similarity_score}
                for c in result.context_window.included_chunks
            ],
        }
        try:
            return await asyncio.to_thread(
                metrics_store.log_query,
                tenant_id=tenant_id, query=body.query, latency_ms=latency_ms,
                tokens_used=result.prompt_tokens + result.completion_tokens, faithfulness_score=None, details=details,
                embedding_tokens=retrieval_result.embedding_tokens, embedding_cost_usd=retrieval_result.embedding_cost_usd,
                generation_input_tokens=result.prompt_tokens, generation_output_tokens=result.completion_tokens,
                rerank_cost_usd=retrieval_result.rerank_cost_usd, provider=result.provider,
                generation_cost_usd=result.generation_cost_usd,
            )
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return None

    @staticmethod
    def evaluate_faithfulness(
        evaluator: FaithfulnessEvaluator, result: GenerationResult, log_id: Optional[int], metrics_store, pipeline_logger
    ) -> Optional[GenerationResult]:
        """Judges the answer against its context and records the score on the query's log row."""
        try:
            evaluated = evaluator.evaluate(result)
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return None
        if log_id and metrics_store:
            try:
                metrics_store.update_faithfulness(log_id, evaluated.faithfulness_score, evaluated.faithfulness_reasoning)
            except Exception as e:
                logger.error(f"Failed to update faithfulness score: {e}")
        if pipeline_logger:
            pipeline_logger.log_event("faithfulness_complete", query_text=result.query, score=evaluated.faithfulness_score)
        return evaluated
