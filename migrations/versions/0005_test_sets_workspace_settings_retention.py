"""Test sets and workspace settings in Postgres, retention indexes, and two unused columns removed.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-03

- test_sets / test_questions: synthetic test sets live in the database (draft, reviewed, frozen)
  instead of JSON files, so the CLI and the API read the same rows.
- workspace_settings: the retrieval configuration a workspace's chat runs.
- timestamp indexes on the tables that are pruned by age (src/maintenance.py).
- documents.visibility and pipeline_events.query_id were never read or written.

Chunk ids became doc-scoped in the same release (src/ingestion/pipeline.py). That needs no schema
change: ids are opaque text. Existing chunks keep their old ids until their document is
re-ingested; the database held no corpus when this migration was written.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NOW = sa.text("now()")
JSON = postgresql.JSONB(astext_type=sa.Text())
EMPTY_ARRAY = sa.text("'[]'::jsonb")


def upgrade() -> None:
    op.drop_column("documents", "visibility")
    op.drop_column("pipeline_events", "query_id")

    op.create_index("ix_fetch_log_created_at", "fetch_log", ["created_at"])
    op.create_index("ix_query_logs_timestamp", "query_logs", ["timestamp"])
    op.create_index("ix_pipeline_events_timestamp", "pipeline_events", ["timestamp"])

    op.create_table(
        "workspace_settings",
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("retrieval_config", JSON, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", name="pk_workspace_settings"),
    )

    op.create_table(
        "test_sets",
        sa.Column("test_set_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("meta", JSON, nullable=False),
        sa.Column("abstained", JSON, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("test_set_id", name="pk_test_sets"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_test_sets_tenant_id_name"),
    )

    op.create_table(
        "test_questions",
        sa.Column("test_set_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), server_default="", nullable=False),
        sa.Column("acceptable_documents", JSON, nullable=False),
        sa.Column("acceptable_headings", JSON, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("source_chunk_ids", JSON, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("difficulty", sa.Text(), server_default="unspecified", nullable=False),
        sa.Column("category", sa.Text(), server_default="unspecified", nullable=False),
        sa.Column("origin", sa.Text(), server_default="manual", nullable=False),
        sa.Column("lexical_overlap", sa.Float(), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.Column("review_status", sa.Text(), server_default="pending", nullable=False),
        sa.ForeignKeyConstraint(
            ["test_set_id"], ["test_sets.test_set_id"], name="fk_test_questions_test_set_id_test_sets", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("test_set_id", "position", name="pk_test_questions"),
    )


def downgrade() -> None:
    op.drop_table("test_questions")
    op.drop_table("test_sets")
    op.drop_table("workspace_settings")
    op.drop_index("ix_pipeline_events_timestamp", table_name="pipeline_events")
    op.drop_index("ix_query_logs_timestamp", table_name="query_logs")
    op.drop_index("ix_fetch_log_created_at", table_name="fetch_log")
    op.add_column("pipeline_events", sa.Column("query_id", sa.Text(), nullable=True))
    op.add_column(
        "documents", sa.Column("visibility", sa.Text(), server_default="private", nullable=False)
    )
