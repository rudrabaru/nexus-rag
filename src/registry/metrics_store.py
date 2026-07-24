import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
from src.observability.costs import (
    JINA_EMBEDDING_COST_PER_TOKEN,
    JINA_RERANK_COST_PER_1K_TOKENS,
    COST_TABLE,
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
    ):
        """Logs query analytics and costs to SQLite."""
        now = datetime.now(timezone.utc).isoformat()


        # Calculate costs
        embedding_cost = embedding_tokens * JINA_EMBEDDING_COST_PER_TOKEN
        
        provider_costs = COST_TABLE.get(provider.lower(), COST_TABLE["gemini"])
        input_cost_rate = provider_costs["input_cost_per_1k"]
        output_cost_rate = provider_costs["output_cost_per_1k"]
        
        generation_cost = (
            (generation_input_tokens / 1000.0) * input_cost_rate +
            (generation_output_tokens / 1000.0) * output_cost_rate
        )
        rerank_cost = (rerank_tokens / 1000.0) * JINA_RERANK_COST_PER_1K_TOKENS
        total_cost = embedding_cost + generation_cost + rerank_cost

        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO observability_logs 
                (tenant_id, timestamp, query, latency_ms, tokens_used, faithfulness_score, details,
                embedding_tokens, embedding_cost_usd, generation_input_tokens, generation_output_tokens,
                generation_cost_usd, rerank_cost_usd, total_cost_usd) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tenant_id, now, query, latency_ms, tokens_used, faithfulness_score, json.dumps(details),
                 embedding_tokens, embedding_cost, generation_input_tokens, generation_output_tokens,
                 generation_cost, rerank_cost, total_cost)
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
