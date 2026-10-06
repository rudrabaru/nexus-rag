"""Query embeddings kept for experiments.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-06

At the card-free Voyage limit (3 requests a minute) embedding the questions of a test set takes a
quarter of an hour, and every rerun, resume or further experiment paid it again, because the
in-memory cache dies with the process. Experiments now keep each query's embedding.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_embeddings",
        sa.Column("index_id", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.Text(), nullable=False),
        sa.Column("embedding", ARRAY(sa.REAL()), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("index_id", "query_hash", name="pk_query_embeddings"),
    )


def downgrade() -> None:
    op.drop_table("query_embeddings")
