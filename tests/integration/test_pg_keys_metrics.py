"""API keys, query metrics and pipeline events."""
import pytest
from sqlalchemy import func, select
from src.observability.logger import PipelineLogger
from src.stores.api_keys import AuthStore
from src.stores.query_log import QueryLogStore
from src.db.schema import pipeline_events, query_logs


pytestmark = pytest.mark.usefixtures("clean_tables")


def test_auth_store_issues_validates_and_revokes_on_postgres(pg_engine):
    store = AuthStore(pg_engine)
    key = store.create_api_key("tenant-1")
    assert store.validate_api_key(key) == "tenant-1"
    assert store.revoke_api_key(key) == 1
    assert store.validate_api_key(key) is None


def test_query_log_persists_provider_cost_and_faithfulness(pg_engine):
    """Regression: the provider argument used to be silently dropped (no column existed)."""
    metrics = QueryLogStore(pg_engine)
    log_id = metrics.log_query(
        tenant_id="tenant-1", query="q", latency_ms=12.5, tokens_used=10, faithfulness_score=None,
        details={"top_k_requested": 5}, provider="groq", generation_cost_usd=0.0042,
    )
    metrics.update_faithfulness(log_id, 0.9, "grounded")

    [row] = metrics.recent_queries("tenant-1")
    assert row["provider"] == "groq"
    assert row["generation_cost_usd"] == pytest.approx(0.0042)
    assert row["faithfulness_score"] == pytest.approx(0.9)
    assert row["details"] == {"top_k_requested": 5, "faithfulness_reasoning": "grounded"}
    assert metrics.recent_queries("tenant-2") == []


def test_pipeline_events_are_persisted_off_the_calling_thread(pg_engine):
    logger = PipelineLogger("test", engine=pg_engine)
    for i in range(3):
        logger.log_event("query_started", tenant_id="tenant-1", query_text=f"q{i}")
    # A longer timeout than the 5s production default: Neon's compute can be cold at the
    # start of a test session (observed: several seconds to resume), and this assertion
    # needs the background writer to have actually flushed, not just been given up on.
    logger.close(timeout=30.0)

    with pg_engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(pipeline_events)).scalar_one() == 3
        assert conn.execute(select(func.count()).select_from(query_logs)).scalar_one() == 0


def test_retention_deletes_only_rows_older_than_each_tables_window(pg_engine):
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import insert

    from src.db.schema import fetch_log, query_logs
    from src.maintenance import RETENTION_DAYS, prune

    now = datetime(2026, 10, 3, tzinfo=timezone.utc)

    def old(table):
        return now - timedelta(days=RETENTION_DAYS[table][2] + 1)

    def fresh(table):
        return now - timedelta(days=RETENTION_DAYS[table][2] - 1)

    with pg_engine.begin() as conn:
        conn.execute(insert(pipeline_events), [{"event": "old", "timestamp": old("pipeline_events")}, {"event": "fresh", "timestamp": fresh("pipeline_events")}])
        conn.execute(insert(query_logs), [{"tenant_id": "t", "query": "old", "timestamp": old("query_logs")}, {"tenant_id": "t", "query": "fresh", "timestamp": fresh("query_logs")}])
        conn.execute(insert(fetch_log), [
            {"tenant_id": "t", "url": "https://x/old", "outcome": "fetched", "created_at": old("fetch_log")},
            {"tenant_id": "t", "url": "https://x/fresh", "outcome": "fetched", "created_at": fresh("fetch_log")},
        ])

    assert prune(pg_engine, now=now) == {"pipeline_events": 1, "query_logs": 1, "fetch_log": 1}

    with pg_engine.connect() as conn:
        assert [r for r in conn.execute(select(pipeline_events.c.event)).scalars()] == ["fresh"]
        assert [r for r in conn.execute(select(query_logs.c.query)).scalars()] == ["fresh"]
        assert [r for r in conn.execute(select(fetch_log.c.url)).scalars()] == ["https://x/fresh"]


def test_a_workspaces_retrieval_settings_are_stored_replaced_and_cleared_per_tenant(pg_engine):
    from src.stores.workspace import WorkspaceSettingsStore

    store = WorkspaceSettingsStore(pg_engine)
    assert store.get_retrieval("tenant-1") is None

    store.put_retrieval("tenant-1", {"strategy": "dense", "rrf_k": 30})
    store.put_retrieval("tenant-1", {"strategy": "sparse"})  # a later choice replaces the earlier one
    store.put_retrieval("tenant-2", {"strategy": "hybrid"})

    assert store.get_retrieval("tenant-1") == {"strategy": "sparse"}
    assert store.get_retrieval("tenant-2") == {"strategy": "hybrid"}
    assert store.clear_retrieval("tenant-1") is True and store.get_retrieval("tenant-1") is None
    assert store.clear_retrieval("tenant-1") is False


def test_usage_totals_cover_the_whole_history_not_just_the_rows_returned(pg_engine):
    store = QueryLogStore(pg_engine)
    for i in range(5):
        store.log_query("tenant-1", f"q{i}", latency_ms=100.0 * (i + 1), tokens_used=10, faithfulness_score=None, details={},
                        generation_cost_usd=0.25)
    store.log_query("tenant-2", "other", latency_ms=1.0, tokens_used=1, faithfulness_score=None, details={}, generation_cost_usd=9.0)

    assert len(store.recent_queries("tenant-1", limit=2)) == 2
    summary = store.summary("tenant-1")
    assert summary["total_queries"] == 5 and summary["total_cost_usd"] == 1.25
    assert summary["avg_cost_per_query_usd"] == 0.25 and summary["avg_latency_ms"] == 300.0
    assert store.summary("nobody") == {"total_queries": 0, "total_cost_usd": 0.0, "avg_cost_per_query_usd": 0.0, "avg_latency_ms": 0.0}


def test_the_system_store_sees_the_database_and_counts_only_recent_worker_heartbeats(pg_engine):
    from sqlalchemy import text

    from src.stores.system import SystemStore

    store = SystemStore(pg_engine)
    assert store.database_ok() is True and store.workers_online() == 0
    with pg_engine.begin() as conn:
        conn.execute(text("INSERT INTO procrastinate_workers (last_heartbeat) VALUES (now()), (now() - interval '10 minutes')"))
    assert store.workers_online() == 1  # the stale heartbeat belongs to a worker that stopped
