"""Embedding checkpoints: embeddings already paid for by a running ingestion.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-03

At the card-free Voyage limit (10K tokens a minute) one ingestion is hours of paced requests. A
retry used to re-embed everything; with checkpoints it resumes where the failed attempt stopped.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC

revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "embedding_checkpoints",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("input_hash", sa.Text(), nullable=False),
        sa.Column("embedding", HALFVEC(1024), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.job_id"], name="fk_embedding_checkpoints_job_id_jobs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("job_id", "chunk_id", name="pk_embedding_checkpoints"),
    )


def downgrade() -> None:
    op.drop_table("embedding_checkpoints")
