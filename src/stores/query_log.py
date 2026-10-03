import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, func, insert, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from src.db.rows import row_to_dict
from src.db.schema import query_logs

logger = logging.getLogger(__name__)


def query_costs(embedding_cost_usd: float, generation_cost_usd: float, rerank_cost_usd: float) -> Dict[str, float]:
    """
    Per-query cost breakdown. Every figure is the caller's per-call value: litellm.completion_cost
    for the model that answered, the query embedder's list price for the index searched, and the
    reranker's own figure (0 for local FlashRank).
    """
    return {
        "embedding_cost_usd": embedding_cost_usd,
        "generation_cost_usd": generation_cost_usd,
        "rerank_cost_usd": rerank_cost_usd,
        "total_cost_usd": embedding_cost_usd + generation_cost_usd + rerank_cost_usd,
    }


class QueryLogStore:
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
        rerank_cost_usd: float = 0.0,
        provider: str = "gemini",
        generation_cost_usd: float = 0.0,
        embedding_cost_usd: float = 0.0,
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
                **query_costs(embedding_cost_usd, generation_cost_usd, rerank_cost_usd),
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

    def summary(self, tenant_id: str) -> Dict[str, Any]:
        """Totals over the tenant's whole retained history (SQL aggregates, not the latest rows)."""
        stmt = select(
            func.count().label("total_queries"),
            func.coalesce(func.sum(query_logs.c.total_cost_usd), 0.0).label("total_cost_usd"),
            func.coalesce(func.avg(query_logs.c.latency_ms), 0.0).label("avg_latency_ms"),
        ).where(query_logs.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().one()
        total = row["total_queries"]
        return {
            "total_queries": total,
            "total_cost_usd": round(float(row["total_cost_usd"]), 6),
            "avg_cost_per_query_usd": round(float(row["total_cost_usd"]) / total, 6) if total else 0.0,
            "avg_latency_ms": round(float(row["avg_latency_ms"]), 2),
        }
