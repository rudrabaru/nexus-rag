"""Embedding indexes (one model per index) and the fetch hand-off and audit tables.

Existing chunks were all embedded by Jina before this revision; each distinct embedding_model
becomes a 'jina:<model>' index and the chunks are assigned to it, so the corpus stays
searchable (EMBEDDING_PROVIDER=jina) until it is re-embedded into a new index.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "embedding_indexes",
        sa.Column("index_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("index_id", name="pk_embedding_indexes"),
    )
    op.execute(
        "INSERT INTO embedding_indexes (index_id, provider, model, dimension) "
        "SELECT DISTINCT 'jina:' || embedding_model, 'jina', embedding_model, 1024 FROM chunks"
    )

    op.add_column("chunks", sa.Column("index_id", sa.Text(), nullable=True))
    op.execute("UPDATE chunks SET index_id = 'jina:' || embedding_model")
    op.alter_column("chunks", "index_id", nullable=False)
    op.create_foreign_key(
        "fk_chunks_index_id_embedding_indexes", "chunks", "embedding_indexes", ["index_id"], ["index_id"]
    )
    op.drop_constraint("pk_chunks", "chunks", type_="primary")
    op.create_primary_key("pk_chunks", "chunks", ["tenant_id", "index_id", "chunk_id"])

    op.create_table(
        "fetched_pages",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.job_id"], name="fk_fetched_pages_job_id_jobs", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "url", name="pk_fetched_pages"),
    )

    op.create_table(
        "fetch_log",
        sa.Column("log_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("log_id", name="pk_fetch_log"),
    )
    op.create_index("ix_fetch_log_tenant_id_created_at", "fetch_log", ["tenant_id", "created_at"])


def downgrade() -> None:
    """Fails if a chunk exists in more than one index (the old key cannot hold both copies)."""
    op.drop_table("fetch_log")
    op.drop_table("fetched_pages")
    op.drop_constraint("pk_chunks", "chunks", type_="primary")
    op.create_primary_key("pk_chunks", "chunks", ["tenant_id", "chunk_id"])
    op.drop_constraint("fk_chunks_index_id_embedding_indexes", "chunks", type_="foreignkey")
    op.drop_column("chunks", "index_id")
    op.drop_table("embedding_indexes")
