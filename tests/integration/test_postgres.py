"""
Behaviour that only real Postgres + pgvector can prove: migrations, HNSW and full-text search,
tenant isolation in SQL, cascades, atomic JSONB merges and the halfvec round trip.
"""

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text

from src.embedding.models import EmbeddedChunk
from src.observability.logger import PipelineLogger
from src.stores.api_keys import AuthStore
from src.stores.tenants import add_embedding_tokens
from tests.integration.helpers import Stores
from src.stores.query_log import QueryLogStore
from src.stores.job_transitions import complete_job
from src.db.schema import EMBEDDING_DIMENSION, chunks, embedding_indexes, pipeline_events, query_logs, tenants
from src.db.schema_version import ALEMBIC_INI, assert_schema_current
from src.retrieving.chunk_store import ChunkStore
from src.retrieving.chunk_writes import existing_source_urls, write_chunks
from src.retrieving.config import RetrievalConfig
from src.retrieving.dense import DenseRetriever
from src.retrieving.pipeline import RetrievalPipeline
from src.retrieving.sparse import SparseRetriever
from src.embedding.embedder import EmbeddingBatch

pytestmark = pytest.mark.usefixtures("clean_tables")


def unit_vector(i: int, j: int = None) -> list:
    """A unit vector along axis i, or between axes i and j, so nearest neighbours are known."""
    v = np.zeros(EMBEDDING_DIMENSION)
    v[i] = 1.0
    if j is not None:
        v[j] = 1.0
    return list(v / np.linalg.norm(v))


TEST_INDEX = "test:test-model"


def chunk(chunk_id, tenant="tenant-1", doc_id="doc-1", chunk_text="hello world", vector=None, index_id=TEST_INDEX):
    return EmbeddedChunk(
        chunk_id=chunk_id, source_url=f"https://example.com/{doc_id}", source_document=f"Doc {doc_id}", title="T",
        heading_path=["Guide", "Setup"], chunk_text=chunk_text, token_count=3, 
        document_version="v", chunk_version="v", tenant_id=tenant, doc_id=doc_id,
        embedding=vector or unit_vector(0), embedding_model=index_id.split(":")[1], index_id=index_id,
    )


@pytest.fixture
def registry(pg_engine):
    return Stores(pg_engine)


@pytest.fixture
def store(pg_engine, pg_async_engine):
    return ChunkStore(pg_engine, pg_async_engine, TEST_INDEX)


@pytest.fixture
def load(pg_engine):
    """Writes chunks in their own transaction, the way src/jobs/commit.py does inside its own."""
    def _load(embedded):
        with pg_engine.begin() as conn:
            write_chunks(conn, embedded)
    return _load


def add_document(registry, doc_id="doc-1", tenant="tenant-1", job_id=None):
    registry.register_job(job_id or f"job-{doc_id}", doc_id, f"https://example.com/{doc_id}", "web", tenant)


# ── Schema ───────────────────────────────────────────────────────────────────

def test_unqualified_names_resolve_to_the_throwaway_schema(pg_engine, test_schema):
    """Guards the isolation every other test depends on (see conftest)."""

    schema_of = text(
        "SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.oid = to_regclass(:t)"
    )
    with pg_engine.connect() as conn:
        for table in ("chunks", "api_keys", "alembic_version"):
            assert conn.execute(schema_of, {"t": table}).scalar() == test_schema, table


def test_database_is_at_the_code_revision(pg_engine):
    assert_schema_current(pg_engine)


def test_migrations_match_the_schema_module(pg_engine):
    """`alembic check`: autogenerate finds no difference between schema.py and the migrated database."""
    config = Config(str(ALEMBIC_INI))
    with pg_engine.connect() as conn:
        config.attributes["connection"] = conn
        command.check(config)


async def test_search_connections_receive_the_hnsw_settings(pg_async_engine):
    """
    Regression: settings sent as individual startup parameters were silently dropped by
    Neon's proxy, so tenant-filtered search would run without iterative scans.
    """

    async with pg_async_engine.connect() as conn:
        iterative = (await conn.execute(text("SELECT current_setting('hnsw.iterative_scan', true)"))).scalar()
        ef_search = (await conn.execute(text("SELECT current_setting('hnsw.ef_search', true)"))).scalar()
    assert (iterative, ef_search) == ("relaxed_order", "100")


# ── Chunk store ──────────────────────────────────────────────────────────────

async def test_dense_search_returns_nearest_first_within_the_tenant(load, registry, store):
    add_document(registry)
    add_document(registry, doc_id="doc-x", tenant="tenant-2")
    load([
        chunk("c-near", vector=unit_vector(0)),
        chunk("c-mid", vector=unit_vector(0, 1)),
        chunk("c-far", vector=unit_vector(5)),
        chunk("other-tenant", tenant="tenant-2", doc_id="doc-x", vector=unit_vector(0)),
    ])

    results = await store.search_dense(unit_vector(0), top_k=10, tenant_id="tenant-1")

    assert [r.chunk_id for r in results] == ["c-near", "c-mid", "c-far"]
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-3)
    assert results[1].similarity_score == pytest.approx(0.7071, abs=1e-3)
    assert results[0].heading_path == ["Guide", "Setup"]


async def test_two_tenants_can_hold_the_same_chunk_id(load, registry, store):
    """Regression: Qdrant keyed points by chunk_id alone, so the second tenant overwrote the first."""
    add_document(registry, doc_id="doc-a", tenant="tenant-1")
    add_document(registry, doc_id="doc-b", tenant="tenant-2")
    load([chunk("same-url_chunk_000", tenant="tenant-1", doc_id="doc-a", chunk_text="tenant one text")])
    load([chunk("same-url_chunk_000", tenant="tenant-2", doc_id="doc-b", chunk_text="tenant two text")])

    one = await store.search_dense(unit_vector(0), top_k=5, tenant_id="tenant-1")
    two = await store.search_dense(unit_vector(0), top_k=5, tenant_id="tenant-2")
    assert [c.text for c in one] == ["tenant one text"]
    assert [c.text for c in two] == ["tenant two text"]


async def test_reloading_a_chunk_replaces_it_in_both_indexes(load, registry, store):
    add_document(registry)
    load([chunk("c1", chunk_text="original wording")])
    load([chunk("c1", chunk_text="replacement wording")])

    assert store.get_collection_size() == 1
    hits, _ = await store.search_sparse("replacement", tenant_id="tenant-1")
    assert [h.chunk_id for h in hits] == ["c1"]
    stale, _ = await store.search_sparse("original", tenant_id="tenant-1")
    assert stale == []


async def test_one_chunk_can_live_in_two_indexes_and_each_store_sees_only_its_own(load, registry, store, pg_engine, pg_async_engine):
    """A corpus re-embedded with another model coexists with the original, for an A/B comparison."""
    add_document(registry)
    other = ChunkStore(pg_engine, pg_async_engine, "voyage:voyage-4")
    load([chunk("c1", chunk_text="shared wording")])
    load([chunk("c1", chunk_text="shared wording", index_id="voyage:voyage-4", vector=unit_vector(3))])

    assert store.get_collection_size() == 1 and other.get_collection_size() == 1
    [hit] = await other.search_dense(unit_vector(3), top_k=5, tenant_id="tenant-1")
    assert hit.metadata["index_id"] == "voyage:voyage-4"
    sparse, _ = await store.search_sparse("shared", tenant_id="tenant-1")
    assert [h.metadata["index_id"] for h in sparse] == [TEST_INDEX]


def test_writing_to_a_new_index_registers_it(load, registry, store, pg_engine):
    add_document(registry)
    load([chunk("c1")])
    with pg_engine.connect() as conn:
        row = conn.execute(select(embedding_indexes).where(embedding_indexes.c.index_id == TEST_INDEX)).mappings().one()
    assert (row["provider"], row["model"], row["dimension"]) == ("test", "test-model", EMBEDDING_DIMENSION)


def test_fetched_pages_round_trip_and_are_discarded_when_the_job_fails(registry):
    add_document(registry, job_id="job-1")
    registry.store_fetched_page("job-1", "https://example.com/b", "B", "# B", "jina")
    registry.store_fetched_page("job-1", "https://example.com/a", "A", "# A", "jina")
    registry.store_fetched_page("job-1", "https://example.com/a", "A2", "# A again", "firecrawl")  # a retried fetch

    assert registry.fetched_urls("job-1") == {"https://example.com/a", "https://example.com/b"}
    pages = {p["url"]: p for p in registry.get_fetched_pages("job-1")}
    assert pages["https://example.com/a"]["markdown"] == "# A again"

    registry.fail_job("job-1", "boom")
    assert registry.get_fetched_pages("job-1") == []


def test_the_daily_quota_counts_only_the_tenants_successful_fetches(registry):
    for outcome in ("fetched", "fetched", "robots_blocked", "failed"):
        registry.log_fetch("tenant-1", "job-1", "https://example.com/x", outcome, "jina")
    registry.log_fetch("tenant-2", "job-2", "https://example.com/y", "fetched", "jina")
    assert registry.pages_fetched_today("tenant-1") == 2


class AxisEmbedder:
    """Embeds every query as unit_vector(0), in the test index; no network."""
    index_id, model = TEST_INDEX, "test-model"

    async def aembed(self, texts, input_type):
        return EmbeddingBatch(vectors=[unit_vector(0) for _ in texts], tokens=len(texts))

    def cost_usd(self, tokens):
        return 0.0


async def test_the_hybrid_pipeline_fuses_real_dense_and_sparse_search_within_one_tenant(load, registry, store):
    add_document(registry)
    add_document(registry, doc_id="doc-x", tenant="tenant-2")
    load([
        chunk("vector-hit", chunk_text="unrelated wording entirely", vector=unit_vector(0)),
        chunk("keyword-hit", chunk_text="rotate the signing keys quarterly", vector=unit_vector(7)),
        chunk("both", chunk_text="signing keys live here", vector=unit_vector(0, 1)),
        chunk("other-tenant", tenant="tenant-2", doc_id="doc-x", chunk_text="signing keys", vector=unit_vector(0)),
    ])
    hybrid = RetrievalPipeline(
        config=RetrievalConfig(strategy="hybrid", top_k=3),
        dense=DenseRetriever(store, AxisEmbedder()), sparse=SparseRetriever(store),
    )

    result = await hybrid.run("signing keys", "tenant-1")

    ids = [c.chunk_id for c in result.chunks]
    assert "other-tenant" not in ids
    assert ids[0] == "both"  # found by both rankings
    assert set(ids) == {"both", "vector-hit", "keyword-hit"} and result.degraded == []


async def test_sparse_search_stems_and_falls_back_from_and_to_or(load, registry, store):
    add_document(registry)
    load([
        chunk("c-both", chunk_text="Configure the firewall rules before running the deployment."),
        chunk("c-one", chunk_text="Firewall basics."),
    ])

    both, fallback = await store.search_sparse("firewall deploy", tenant_id="tenant-1")
    assert [c.chunk_id for c in both] == ["c-both"] and fallback is False  # 'deploy' matches 'deployment'

    either, fallback = await store.search_sparse("firewall kubernetes", tenant_id="tenant-1")
    assert {c.chunk_id for c in either} == {"c-both", "c-one"} and fallback is True


async def test_sparse_search_treats_query_syntax_as_plain_text(load, registry, store):
    add_document(registry)
    load([chunk("c1", chunk_text="plain text")])
    for hostile in ["a & | ! (", "'; DROP TABLE chunks; --", ":* <-> !!"]:
        await store.search_sparse(hostile, tenant_id="tenant-1")
    assert store.get_collection_size() == 1


def test_halfvec_round_trip_preserves_direction(load, registry, store, pg_engine):
    add_document(registry)
    original = np.random.default_rng(0).normal(size=EMBEDDING_DIMENSION)
    original /= np.linalg.norm(original)
    load([chunk("c1", vector=list(original))])

    with pg_engine.connect() as conn:
        stored = np.asarray(conn.execute(select(chunks.c.embedding)).scalar_one(), dtype=np.float64)
    assert float(original @ stored / np.linalg.norm(stored)) > 0.9999


def test_existing_urls_are_scoped_to_tenant_and_document(load, registry, pg_engine):
    add_document(registry, doc_id="doc-1")
    add_document(registry, doc_id="doc-2")
    load([chunk("a", doc_id="doc-1"), chunk("b", doc_id="doc-2")])
    with pg_engine.connect() as conn:
        assert existing_source_urls(conn, "tenant-1", TEST_INDEX, "doc-1") == {"https://example.com/doc-1"}
        assert existing_source_urls(conn, "tenant-2", TEST_INDEX, "doc-1") == set()


# ── Registry ─────────────────────────────────────────────────────────────────

def test_deleting_a_document_cascades_to_its_chunks_and_jobs(load, registry, store):
    add_document(registry)
    load([chunk("c1"), chunk("c2")])
    assert registry.get_document("doc-1")["chunk_count"] == 2

    assert registry.delete_document("doc-1") is True
    assert store.get_collection_size() == 0
    assert registry.get_job("job-doc-1") is None
    assert registry.delete_document("doc-1") is False


def test_job_lifecycle_and_atomic_metadata_merge(load, registry, pg_engine):
    add_document(registry)
    registry.update_job_status("job-doc-1", "processing", 10, metadata={"total_pages": 5})
    registry.update_job_status("job-doc-1", "processing", 50, metadata={"indexed_pages": 3})
    load([chunk("c1")])
    with pg_engine.begin() as conn:
        complete_job(conn, "job-doc-1", {"total_tokens": 42}, status="partial_success")

    job = registry.get_job("job-doc-1")
    assert job["metadata"] == {"total_pages": 5, "indexed_pages": 3}
    assert job["status"] == "partial_success" and job["progress_pct"] == 100
    assert job["error"]  # partial success always explains itself

    doc = registry.get_document("doc-1")
    assert doc["status"] == "partial_success"
    assert doc["stats"] == {"total_tokens": 42}
    assert doc["chunk_count"] == 1
    assert isinstance(doc["ingested_at"], str)


def test_re_registering_a_complete_document_keeps_it_complete(registry, pg_engine):
    add_document(registry, job_id="j1")
    with pg_engine.begin() as conn:
        complete_job(conn, "j1", {})
    registry.register_job("j2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    assert registry.get_document("doc-1")["status"] == "complete"


def test_fail_job_marks_the_job_and_document_failed_and_discards_any_pending_upload(registry):
    add_document(registry)
    registry.update_job_status("job-doc-1", "processing", 40)
    registry.fail_job("job-doc-1", "boom")
    assert registry.get_job("job-doc-1")["status"] == "failed"
    assert registry.get_job("job-doc-1")["error"] == "boom"
    assert registry.get_document("doc-1")["status"] == "failed"


def test_a_duplicate_url_ingestion_completes_against_the_existing_document(load, registry):
    """
    Regression: a fetched page whose content matched an indexed document was marked complete,
    but its placeholder document stayed "pending" forever and its fetched pages were never deleted.
    """
    add_document(registry, doc_id="doc-orig", job_id="j-orig")
    load([chunk("c1", doc_id="doc-orig")])
    add_document(registry, doc_id="doc-dup", job_id="j-dup")
    registry.store_fetched_page("j-dup", "https://example.com/mirror", "M", "# same text", "jina")

    registry.complete_as_duplicate("j-dup", "doc-orig")

    job = registry.get_job("j-dup")
    assert (job["doc_id"], job["status"], job["progress_pct"]) == ("doc-orig", "complete", 100)
    assert job["metadata"] == {"duplicate_of": "doc-orig"}
    assert registry.get_document("doc-dup") is None
    assert registry.get_fetched_pages("j-dup") == []
    assert registry.get_document("doc-orig")["chunk_count"] == 1


def test_a_duplicate_keeps_a_placeholder_that_already_holds_chunks(load, registry):
    """A resumed document with chunks of its own is not deleted as a placeholder."""
    add_document(registry, doc_id="doc-orig", job_id="j-orig")
    add_document(registry, doc_id="doc-resumed", job_id="j-resume")
    load([chunk("c1", doc_id="doc-resumed")])

    registry.complete_as_duplicate("j-resume", "doc-orig")

    assert registry.get_document("doc-resumed")["chunk_count"] == 1


def test_quota_and_counts_are_per_tenant(load, registry, store):
    add_document(registry, doc_id="doc-1", tenant="tenant-1")
    add_document(registry, doc_id="doc-2", tenant="tenant-2")
    load([chunk("a"), chunk("b"), chunk("c", tenant="tenant-2", doc_id="doc-2")])
    assert registry.chunk_count("tenant-1") == 2
    assert registry.document_count("tenant-2") == 1
    assert [d["doc_id"] for d in registry.list_documents("tenant-2")] == ["doc-2"]
    assert len(registry.list_all_documents()) == 2


def test_tenant_token_usage_accumulates(pg_engine):
    for tokens in (100, 50, 0):
        with pg_engine.begin() as conn:
            add_embedding_tokens(conn, "tenant-1", tokens)
    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.total_embedding_tokens)).scalar_one() == 150


# ── Keys, metrics, events ────────────────────────────────────────────────────

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
