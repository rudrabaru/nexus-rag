"""The two best-effort side effects of an answered chat query: its log row and its faithfulness score. Neither ever raises."""
import asyncio
import logging
from typing import Optional

from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.models import GenerationResult
from src.services.chat_models import Prepared
from src.stores.query_log import QueryLogStore

logger = logging.getLogger(__name__)


async def record_query(
    query_log: Optional[QueryLogStore], prepared: Prepared, result: GenerationResult, latency_ms: float
) -> Optional[int]:
    """Persists the query's latency, tokens, serving provider and cost; returns the log row id."""
    if not query_log:
        return None
    retrieval = prepared.retrieval
    details = {
        "top_k_requested": prepared.chat.top_k,
        "retrieved_context": [
            {"chunk_id": c.chunk_id, "source_url": c.source_url, "similarity_score": c.similarity_score}
            for c in result.context_window.included_chunks
        ],
    }
    try:
        return await asyncio.to_thread(
            query_log.log_query,
            tenant_id=prepared.tenant_id, query=prepared.chat.query, latency_ms=latency_ms,
            tokens_used=result.prompt_tokens + result.completion_tokens, faithfulness_score=None, details=details,
            embedding_tokens=retrieval.embedding_tokens, embedding_cost_usd=retrieval.embedding_cost_usd,
            generation_input_tokens=result.prompt_tokens, generation_output_tokens=result.completion_tokens,
            rerank_cost_usd=retrieval.rerank_cost_usd, provider=result.provider,
            generation_cost_usd=result.generation_cost_usd,
        )
    except Exception:
        logger.exception("Failed to log the query")
        return None


def judge_answer(
    evaluator: FaithfulnessEvaluator, query_log: Optional[QueryLogStore], result: GenerationResult, log_id: Optional[int]
) -> Optional[GenerationResult]:
    """Judges the answer against its context and records the score on the query's log row."""
    try:
        judged = evaluator.evaluate(result)
    except Exception:
        logger.exception("Faithfulness evaluation failed")
        return None
    if log_id and query_log:
        try:
            query_log.update_faithfulness(log_id, judged.faithfulness_score, judged.faithfulness_reasoning)
        except Exception:
            logger.exception("Failed to record the faithfulness score")
    return judged
