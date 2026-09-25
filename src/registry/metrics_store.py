import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, func, insert, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from src.observability.costs import JINA_EMBEDDING_COST_PER_TOKEN, JINA_RERANK_COST_PER_1K_TOKENS
from src.registry.rows import row_to_dict
from src.registry.schema import query_logs

logger = logging.getLogger(__name__)


def query_costs(embedding_tokens: int, generation_cost_usd: float, rerank_tokens: int) -> Dict[str, float]:
    """
    Per-query cost breakdown. Generation cost is the caller's real per-call figure
    (litellm.completion_cost for the model that actually answered, fallbacks included), not
    re-derived here from a rate table. Embedding and rerank still use Jina's fixed rates.
    """
    embedding = embedding_tokens * JINA_EMBEDDING_COST_PER_TOKEN
    rerank = (rerank_tokens / 1000.0) * JINA_RERANK_COST_PER_1K_TOKENS
    return {
        "embedding_cost_usd": embedding,
        "generation_cost_usd": generation_cost_usd,
        "rerank_cost_usd": rerank,
        "total_cost_usd": embedding + generation_cost_usd + rerank,
    }


class MetricsStore:
    """Per-query analytics and cost, stored in Postgres."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def log_query(
        self,
        tenant_id: str,
        query: str,
        latency_ms: float,
        tokens_used: int,
        faithfulness_score: Optional[float],
        details: Dict[str, Any],
        embedding_tokens: int = 0,
        generation_input_tokens: int = 0,
        generation_output_tokens: int = 0,
        rerank_tokens: int = 0,
        provider: str = "gemini",
        generation_cost_usd: float = 0.0,
    ) -> int:
        stmt = (
            insert(query_logs)
            .values(
                tenant_id=tenant_id,
                query=query,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                faithfulness_score=faithfulness_score,
                details=details,
                provider=provider,
                embedding_tokens=embedding_tokens,
                generation_input_tokens=generation_input_tokens,
                generation_output_tokens=generation_output_tokens,
                **query_costs(embedding_tokens, generation_cost_usd, rerank_tokens),
            )
            .returning(query_logs.c.log_id)
        )
        with self._engine.begin() as conn:
            return conn.execute(stmt).scalar_one()

    def update_faithfulness(self, log_id: int, score: float, reasoning: str) -> None:
        """Records a background faithfulness evaluation on an existing log row."""
        reasoning_patch = bindparam("patch", {"faithfulness_reasoning": reasoning}, type_=JSONB)
        stmt = (
            update(query_logs)
            .where(query_logs.c.log_id == log_id)
            .values(
                faithfulness_score=score,
                details=func.coalesce(query_logs.c.details, bindparam("empty", {}, type_=JSONB)).op("||")(reasoning_patch),
            )
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def recent_queries(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        stmt = (
            select(query_logs)
            .where(query_logs.c.tenant_id == tenant_id)
            .order_by(query_logs.c.log_id.desc())
            .limit(limit)
        )
        with self._engine.connect() as conn:
            return [row_to_dict(row) for row in conn.execute(stmt)]
