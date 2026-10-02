"""Evaluation engine: experiments, trials, per-query runs, and the generation and judge caches.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NOW = sa.text("now()")
JSON = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("experiment_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("spec", JSON, nullable=False),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("dataset_hash", sa.Text(), nullable=False),
        sa.Column("queries", JSON, nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("summary", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("experiment_id", name="pk_experiments"),
    )
    op.create_index("ix_experiments_tenant_id_created_at", "experiments", ["tenant_id", "created_at"])

    op.create_table(
        "trials",
        sa.Column("trial_id", sa.Text(), nullable=False),
        sa.Column("experiment_id", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("config", JSON, nullable=False),
        sa.Column("index_id", sa.Text(), nullable=False),
        sa.Column("index_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["experiments.experiment_id"], name="fk_trials_experiment_id_experiments", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("trial_id", name="pk_trials"),
        sa.UniqueConstraint("experiment_id", "label", name="uq_trials_experiment_id_label"),
    )

    op.create_table(
        "runs",
        sa.Column("trial_id", sa.Text(), nullable=False),
        sa.Column("query_index", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("exact_rank", sa.Integer(), nullable=True),
        sa.Column("first_stage_rank", sa.Integer(), nullable=True),
        sa.Column("retrieved", JSON, nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("embedding_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("embedding_cost_usd", sa.Float(), server_default="0", nullable=False),
        sa.Column("rerank_cost_usd", sa.Float(), server_default="0", nullable=False),
        sa.Column("degraded", JSON, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("generation_model", sa.Text(), nullable=True),
        sa.Column("generation_input_tokens", sa.Integer(), nullable=True),
        sa.Column("generation_output_tokens", sa.Integer(), nullable=True),
        sa.Column("generation_cost_usd", sa.Float(), nullable=True),
        sa.Column("generation_cached", sa.Boolean(), nullable=True),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("faithfulness_reasoning", sa.Text(), nullable=True),
        sa.Column("judge_model", sa.Text(), nullable=True),
        sa.Column("judge_cached", sa.Boolean(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(["trial_id"], ["trials.trial_id"], name="fk_runs_trial_id_trials", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("trial_id", "query_index", name="pk_runs"),
    )

    op.create_table(
        "generation_cache",
        sa.Column("cache_key", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("cache_key", name="pk_generation_cache"),
    )

    op.create_table(
        "judge_cache",
        sa.Column("cache_key", sa.Text(), nullable=False),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("judge_model", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("cache_key", name="pk_judge_cache"),
    )


def downgrade() -> None:
    op.drop_table("judge_cache")
    op.drop_table("generation_cache")
    op.drop_table("runs")
    op.drop_table("trials")
    op.drop_index("ix_experiments_tenant_id_created_at", table_name="experiments")
    op.drop_table("experiments")
