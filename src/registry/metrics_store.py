import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
from src.observability.costs import (
    JINA_EMBEDDING_COST_PER_TOKEN,
    JINA_RERANK_COST_PER_1K_TOKENS,
)

logger = logging.getLogger(__name__)

class MetricsStore:
    """
    Manages observability and query metrics logging using the shared registry database.
    """
    def __init__(self, get_conn_func):
        self._get_conn = get_conn_func

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
    ):
        """
        Logs query analytics and costs to SQLite. generation_cost_usd is computed by the
        caller from the LLM client's actual per-call cost (litellm.completion_cost against
        the real model used, including whichever provider a fallback landed on) rather
        than being re-derived here from token counts and a static rate table.
        """
        now = datetime.now(timezone.utc).isoformat()

        embedding_cost = embedding_tokens * JINA_EMBEDDING_COST_PER_TOKEN
        generation_cost = generation_cost_usd
        rerank_cost = (rerank_tokens / 1000.0) * JINA_RERANK_COST_PER_1K_TOKENS
        total_cost = embedding_cost + generation_cost + rerank_cost

        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO observability_logs
                (tenant_id, timestamp, query, latency_ms, tokens_used, faithfulness_score, details,
                embedding_tokens, embedding_cost_usd, generation_input_tokens, generation_output_tokens,
                generation_cost_usd, rerank_cost_usd, total_cost_usd, provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tenant_id, now, query, latency_ms, tokens_used, faithfulness_score, json.dumps(details),
                 embedding_tokens, embedding_cost, generation_input_tokens, generation_output_tokens,
                 generation_cost, rerank_cost, total_cost, provider)
            )
            log_id = cursor.lastrowid
            conn.commit()
            return log_id

    def update_faithfulness(self, log_id: int, score: float, reasoning: str):
        """Updates an existing log with background faithfulness evaluation results."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT details FROM observability_logs WHERE log_id = ?", (log_id,))
            row = cursor.fetchone()
            if not row:
                return
                
            details = json.loads(row["details"]) if row["details"] else {}
            details["faithfulness_reasoning"] = reasoning
            
            conn.execute(
                "UPDATE observability_logs SET faithfulness_score = ?, details = ? WHERE log_id = ?",
                (score, json.dumps(details), log_id)
            )
            conn.commit()
