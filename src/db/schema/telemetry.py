"""What the service records about itself: per-query cost and latency, and pipeline events. Both are pruned (src/maintenance.py)."""
from sqlalchemy import BigInteger, Column, DateTime, Float, Identity, Index, Integer, Table, Text

from src.db.schema.base import Json, metadata, now

query_logs = Table(
    "query_logs",
    metadata,
    Column("log_id", BigInteger, Identity(), primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False, server_default=now()),
    Column("query", Text, nullable=False),
    Column("latency_ms", Float),
    Column("tokens_used", Integer),
    Column("faithfulness_score", Float),
    Column("details", Json),
    Column("provider", Text),
    Column("embedding_tokens", Integer),
    Column("embedding_cost_usd", Float),
    Column("generation_input_tokens", Integer),
    Column("generation_output_tokens", Integer),
    Column("generation_cost_usd", Float),
    Column("rerank_cost_usd", Float),
    Column("total_cost_usd", Float),
    Index(None, "tenant_id", "log_id"),
    Index(None, "timestamp"),
)

pipeline_events = Table(
    "pipeline_events",
    metadata,
    Column("event_id", BigInteger, Identity(), primary_key=True),
    Column("event", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False, server_default=now()),
    Column("tenant_id", Text),
    Column("job_id", Text),
    Column("details", Json),
    Index(None, "timestamp"),
)
