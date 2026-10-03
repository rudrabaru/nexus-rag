"""
Evaluation (src/evaluation, src/testsets): test sets, experiments and what a workspace has chosen.

- A test set is a list of questions with their ground truth. It is a draft while questions are
  generated and reviewed, and frozen (immutable, hashed) when an experiment may use it.
- An experiment freezes its query set; it has one trial per retrieval configuration and one run
  per trial and query. Runs are the per-query evidence the significance tests need, so they are
  stored, not only aggregated. The two caches hold LLM answers and judge scores keyed by a hash
  of everything that determines them.
- workspace_settings holds the retrieval configuration a workspace chat runs (the winner an
  experiment promoted); a workspace without a row uses the environment's defaults.
"""
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    text,
)

from src.db.schema.base import Json, metadata, now

workspace_settings = Table(
    "workspace_settings",
    metadata,
    Column("tenant_id", Text, primary_key=True),
    Column("retrieval_config", Json, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=now()),
)

test_sets = Table(
    "test_sets",
    metadata,
    Column("test_set_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("status", Text, nullable=False),  # draft | frozen
    Column("meta", Json, nullable=False),  # how it was generated: index, model, seed, difficulties, temperature
    Column("abstained", Json, nullable=False, server_default=text("'[]'::jsonb")),  # chunks the model found nothing to ask about
    Column("content_hash", Text),  # of the accepted questions; set when frozen
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
    Column("frozen_at", DateTime(timezone=True)),
    UniqueConstraint("tenant_id", "name"),
)

test_questions = Table(
    "test_questions",
    metadata,
    Column("test_set_id", Text, ForeignKey("test_sets.test_set_id", ondelete="CASCADE"), nullable=False),
    Column("position", Integer, nullable=False),
    Column("query", Text, nullable=False),
    Column("reference_answer", Text, nullable=False, server_default=""),
    Column("acceptable_documents", Json, nullable=False),
    Column("acceptable_headings", Json, nullable=False, server_default=text("'[]'::jsonb")),
    Column("source_chunk_ids", Json, nullable=False, server_default=text("'[]'::jsonb")),
    Column("difficulty", Text, nullable=False, server_default="unspecified"),
    Column("category", Text, nullable=False, server_default="unspecified"),
    Column("origin", Text, nullable=False, server_default="manual"),  # "synthetic" for generated questions
    Column("lexical_overlap", Float),
    Column("source_text", Text),
    Column("review_status", Text, nullable=False, server_default="pending"),  # pending | accepted | rejected
    PrimaryKeyConstraint("test_set_id", "position"),
)

experiments = Table(
    "experiments",
    metadata,
    Column("experiment_id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("tenant_id", Text, nullable=False),
    Column("spec", Json, nullable=False),
    Column("dataset_name", Text, nullable=False),
    Column("dataset_hash", Text, nullable=False),
    Column("queries", Json, nullable=False),  # the frozen query set: a dataset edited later cannot change it
    Column("status", Text, nullable=False),  # running | paused | complete | failed
    Column("summary", Json),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
    Column("finished_at", DateTime(timezone=True)),
    Index(None, "tenant_id", "created_at"),
)

trials = Table(
    "trials",
    metadata,
    Column("trial_id", Text, primary_key=True),
    Column("experiment_id", Text, ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False),
    Column("label", Text, nullable=False),
    Column("config", Json, nullable=False),  # RetrievalConfig, every knob
    Column("index_id", Text, nullable=False),
    Column("index_count", Integer, nullable=False),  # chunks in the index when the trial started
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
    UniqueConstraint("experiment_id", "label"),
)

runs = Table(
    "runs",
    metadata,
    Column("trial_id", Text, ForeignKey("trials.trial_id", ondelete="CASCADE"), nullable=False),
    Column("query_index", Integer, nullable=False),
    Column("rank", Integer),  # 1-based rank of the first relevant chunk; NULL = none retrieved
    Column("exact_rank", Integer),  # same, counting only chunks whose heading also matched
    Column("first_stage_rank", Integer),  # before reranking (reranked trials only)
    Column("retrieved", Json, nullable=False),  # [{chunk_id, source, score, match}] in rank order
    Column("latency_ms", Float, nullable=False),
    Column("embedding_tokens", Integer, nullable=False, server_default="0"),  # as if uncached
    Column("embedding_cost_usd", Float, nullable=False, server_default="0"),
    Column("rerank_cost_usd", Float, nullable=False, server_default="0"),
    Column("degraded", Json, nullable=False, server_default=text("'[]'::jsonb")),
    Column("answer", Text),
    Column("generation_model", Text),
    Column("generation_input_tokens", Integer),
    Column("generation_output_tokens", Integer),
    Column("generation_cost_usd", Float),
    Column("generation_cached", Boolean),
    Column("faithfulness", Float),
    Column("faithfulness_reasoning", Text),
    Column("judge_model", Text),
    Column("judge_cached", Boolean),
    Column("error", Text),  # set = this run is invalid (generation failed) and is retried on resume
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
    PrimaryKeyConstraint("trial_id", "query_index"),
)

generation_cache = Table(
    "generation_cache",
    metadata,
    Column("cache_key", Text, primary_key=True),  # sha256(tenant, model, prompt)
    Column("model", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("input_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("cost_usd", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
)

judge_cache = Table(
    "judge_cache",
    metadata,
    Column("cache_key", Text, primary_key=True),  # sha256(tenant, metric, judge model, question, answer, context ids)
    Column("metric", Text, nullable=False),
    Column("judge_model", Text, nullable=False),
    Column("score", Float, nullable=False),
    Column("reasoning", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
)
