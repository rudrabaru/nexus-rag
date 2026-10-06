"""Initial Postgres schema: replaces Qdrant + the SQLite registry and FTS5 index.

Revision ID: 0001
Revises:
Create Date: 2026-09-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import HALFVEC

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NOW = sa.text("now()")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_embedding_tokens", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", name="pk_tenants"),
    )

    op.create_table(
        "api_keys",
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("key_hash", name="pk_api_keys"),
    )
    op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])

    op.create_table(
        "documents",
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("format", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="private", nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("stats", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("doc_id", name="pk_documents"),
    )
    op.create_index("ix_documents_tenant_id_content_hash", "documents", ["tenant_id", "content_hash"])
    op.create_index("ix_documents_tenant_id_source", "documents", ["tenant_id", "source"])

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("progress_pct", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(
            ["doc_id"], ["documents.doc_id"], name="fk_jobs_doc_id_documents", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("job_id", name="pk_jobs"),
    )
    op.create_index("ix_jobs_doc_id", "jobs", ["doc_id"])

    op.create_table(
        "chunks",
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("source_document", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("section_title", sa.Text(), nullable=True),
        sa.Column("heading_path", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=True),
        sa.Column("contains_code", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("contains_table", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("chunk_version", sa.Text(), nullable=True),
        sa.Column("document_version", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", HALFVEC(1024), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', chunk_text)", persisted=True),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["doc_id"], ["documents.doc_id"], name="fk_chunks_doc_id_documents", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("tenant_id", "chunk_id", name="pk_chunks"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "halfvec_cosine_ops"},
    )
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"], postgresql_using="gin")

    op.create_table(
        "query_logs",
        sa.Column("log_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("embedding_tokens", sa.Integer(), nullable=True),
        sa.Column("embedding_cost_usd", sa.Float(), nullable=True),
        sa.Column("generation_input_tokens", sa.Integer(), nullable=True),
        sa.Column("generation_output_tokens", sa.Integer(), nullable=True),
        sa.Column("generation_cost_usd", sa.Float(), nullable=True),
        sa.Column("rerank_cost_usd", sa.Float(), nullable=True),
        sa.Column("total_cost_usd", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("log_id", name="pk_query_logs"),
    )
    op.create_index("ix_query_logs_tenant_id_log_id", "query_logs", ["tenant_id", "log_id"])

    op.create_table(
        "pipeline_events",
        sa.Column("event_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("query_id", sa.Text(), nullable=True),
        sa.Column("job_id", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("event_id", name="pk_pipeline_events"),
    )


def downgrade() -> None:
    for table in ("pipeline_events", "query_logs", "chunks", "jobs", "documents", "api_keys", "tenants"):
        op.drop_table(table)
