"""Evaluation integrity: what a run records so its numbers can be interpreted.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-04

- latency without the query embedding: the first trial of an experiment pays for embedding each
  query and later trials hit the cache, so total latency made the first trial look slowest. Both
  parts are recorded, and reports compare the part that does not depend on cache order.
- the judge's cost, which was discarded, and whether a run had no context (no answer, no score).
- exact_rank is dropped: nothing ever read it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("embedding_latency_ms", sa.Float(), nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("judge_cost_usd", sa.Float(), nullable=True))
    op.add_column("runs", sa.Column("empty_context", sa.Boolean(), nullable=True))
    op.drop_column("runs", "exact_rank")


def downgrade() -> None:
    op.add_column("runs", sa.Column("exact_rank", sa.Integer(), nullable=True))
    op.drop_column("runs", "empty_context")
    op.drop_column("runs", "judge_cost_usd")
    op.drop_column("runs", "embedding_latency_ms")
