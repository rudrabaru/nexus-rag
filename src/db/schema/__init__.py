"""
The Postgres schema. One MetaData object is shared by the sync engine (stores, jobs, keys,
metrics), the async engine (query-time search) and Alembic (migrations).

Tables are grouped by what they are for, one module each:

    tenancy      tenants, api_keys
    ingestion    documents, jobs, ingest_sources, fetched_pages, fetch_log
    index        embedding_indexes, chunks
    telemetry    query_logs, pipeline_events
    evaluation   workspace_settings, test_sets, test_questions, experiments, trials, runs, caches

Importing this package imports every module, so `metadata` always holds the whole schema.
"""
from src.db.schema.base import EMBEDDING_DIMENSION, TEXT_SEARCH_CONFIG, Json, metadata
from src.db.schema.evaluation import (
    experiments,
    generation_cache,
    judge_cache,
    runs,
    test_questions,
    test_sets,
    trials,
    workspace_settings,
)
from src.db.schema.index import chunks, embedding_indexes
from src.db.schema.ingestion import documents, fetch_log, fetched_pages, ingest_sources, jobs
from src.db.schema.telemetry import pipeline_events, query_logs
from src.db.schema.tenancy import api_keys, tenants

__all__ = [
    "EMBEDDING_DIMENSION", "TEXT_SEARCH_CONFIG", "Json", "metadata",
    "api_keys", "chunks", "documents", "embedding_indexes", "experiments", "fetch_log", "fetched_pages",
    "generation_cache", "ingest_sources", "jobs", "judge_cache", "pipeline_events", "query_logs", "runs",
    "tenants", "test_questions", "test_sets", "trials", "workspace_settings",
]
