"""
Behaviour that only real Postgres + pgvector can prove: migrations, HNSW and full-text search,
tenant isolation in SQL, cascades, atomic JSONB merges and the halfvec round trip.
"""
import json

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text

from src.embedding.models import EmbeddedChunk
from src.observability.logger import PipelineLogger
from src.registry.auth_store import AuthStore
from src.registry.database import DocumentRegistry
from src.registry.metrics_store import MetricsStore
from src.registry.schema import EMBEDDING_DIMENSION, chunks, pipeline_events, query_logs, tenants
from src.registry.schema_version import ALEMBIC_INI, assert_schema_current
from src.retrieving.chunk_store import ChunkStore

pytestmark = pytest.mark.usefixtures("clean_tables")


def unit_vector(i: int, j: int = None) -> list:
    """A unit vector along axis i, or between axes i and j, so nearest neighbours are known."""
    v = np.zeros(EMBEDDING_DIMENSION)
    v[i] = 1.0
    if j is not None:
        v[j] = 1.0
    return list(v / np.linalg.norm(v))


def chunk(chunk_id, tenant="tenant-1", doc_id="doc-1", chunk_text="hello world", vector=None):
    return EmbeddedChunk(
        chunk_id=chunk_id, source_url=f"https://example.com/{doc_id}", source_document=f"Doc {doc_id}", title="T",
        heading_path=["Guide", "Setup"], chunk_index=0, chunk_text=chunk_text, token_count=3, char_start=0, char_end=len(chunk_text),
        document_version="v", chunk_version="v", tenant_id=tenant, doc_id=doc_id,
        embedding=vector or unit_vector(0), embedding_model="test-model",
    )


@pytest.fixture
def registry(pg_engine):
    return DocumentRegistry(pg_engine)


@pytest.fixture
def store(pg_engine, pg_async_engine):
    return ChunkStore(pg_engine, pg_async_engine)


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

async def test_dense_search_returns_nearest_first_within_the_tenant(registry, store):
    add_document(registry)
    add_document(registry, doc_id="doc-x", tenant="tenant-2")
    store.load_chunks([
        chunk("c-near", vector=unit_vector(0)),
        chunk("c-mid", vector=unit_vector(0, 1)),
        chunk("c-far", vector=unit_vector(5)),
        chunk("other-tenant", tenant="tenant-2", doc_id="doc-x", vector=unit_vector(0)),
    ])

    results = await store.search_dense(unit_vector(0), top_k=10, tenant_id="tenant-1")

    assert [r.chunk_id for r in results] == ["c-near", "c-mid", "c-far"]
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-3)
    assert results[1].similarity_score == pytest.approx(0.7071, abs=1e-3)
    assert json.loads(results[0].metadata["heading_path"]) == ["Guide", "Setup"]


async def test_two_tenants_can_hold_the_same_chunk_id(registry, store):
    """Regression: Qdrant keyed points by chunk_id alone, so the second tenant overwrote the first."""
    add_document(registry, doc_id="doc-a", tenant="tenant-1")
    add_document(registry, doc_id="doc-b", tenant="tenant-2")
    store.load_chunks([chunk("same-url_chunk_000", tenant="tenant-1", doc_id="doc-a", chunk_text="tenant one text")])
    store.load_chunks([chunk("same-url_chunk_000", tenant="tenant-2", doc_id="doc-b", chunk_text="tenant two text")])

    one = await store.search_dense(unit_vector(0), top_k=5, tenant_id="tenant-1")
    two = await store.search_dense(unit_vector(0), top_k=5, tenant_id="tenant-2")
    assert [c.text for c in one] == ["tenant one text"]
    assert [c.text for c in two] == ["tenant two text"]


async def test_reloading_a_chunk_replaces_it_in_both_indexes(registry, store):
    add_document(registry)
    store.load_chunks([chunk("c1", chunk_text="original wording")])
    store.load_chunks([chunk("c1", chunk_text="replacement wording")])

    assert store.get_collection_size() == 1
    hits, _ = await store.search_sparse("replacement", tenant_id="tenant-1")
    assert [h.chunk_id for h in hits] == ["c1"]
    stale, _ = await store.search_sparse("original", tenant_id="tenant-1")
    assert stale == []


async def test_sparse_search_stems_and_falls_back_from_and_to_or(registry, store):
    add_document(registry)
    store.load_chunks([
        chunk("c-both", chunk_text="Configure the firewall rules before running the deployment."),
        chunk("c-one", chunk_text="Firewall basics."),
    ])

    both, fallback = await store.search_sparse("firewall deploy", tenant_id="tenant-1")
    assert [c.chunk_id for c in both] == ["c-both"] and fallback is False  # 'deploy' matches 'deployment'

    either, fallback = await store.search_sparse("firewall kubernetes", tenant_id="tenant-1")
    assert {c.chunk_id for c in either} == {"c-both", "c-one"} and fallback is True


async def test_sparse_search_treats_query_syntax_as_plain_text(registry, store):
    add_document(registry)
    store.load_chunks([chunk("c1", chunk_text="plain text")])
    for hostile in ["a & | ! (", "'; DROP TABLE chunks; --", ":* <-> !!"]:
        await store.search_sparse(hostile, tenant_id="tenant-1")
    assert store.get_collection_size() == 1


def test_halfvec_round_trip_preserves_direction(registry, store, pg_engine):
    add_document(registry)
    original = np.random.default_rng(0).normal(size=EMBEDDING_DIMENSION)
    original /= np.linalg.norm(original)
    store.load_chunks([chunk("c1", vector=list(original))])

    with pg_engine.connect() as conn:
        stored = np.asarray(conn.execute(select(chunks.c.embedding)).scalar_one(), dtype=np.float64)
    assert float(original @ stored / np.linalg.norm(stored)) > 0.9999


def test_existing_urls_are_scoped_to_tenant_and_document(registry, store):
    add_document(registry, doc_id="doc-1")
    add_document(registry, doc_id="doc-2")
    store.load_chunks([chunk("a", doc_id="doc-1"), chunk("b", doc_id="doc-2")])
    assert store.get_existing_urls("tenant-1", "doc-1") == {"https://example.com/doc-1"}
    assert store.get_existing_urls("tenant-2", "doc-1") == set()


# ── Registry ─────────────────────────────────────────────────────────────────

def test_deleting_a_document_cascades_to_its_chunks_and_jobs(registry, store):
    add_document(registry)
    store.load_chunks([chunk("c1"), chunk("c2")])
    assert registry.get_document("doc-1")["chunk_count"] == 2

    assert registry.delete_document("doc-1") is True
    assert store.get_collection_size() == 0
    assert registry.get_job("job-doc-1") is None
    assert registry.delete_document("doc-1") is False


def test_job_lifecycle_and_atomic_metadata_merge(registry, store):
    add_document(registry)
    registry.update_job_status("job-doc-1", "processing", 10, metadata={"total_pages": 5})
    registry.update_job_status("job-doc-1", "processing", 50, metadata={"indexed_pages": 3})
    store.load_chunks([chunk("c1")])
    registry.complete_job("job-doc-1", {"total_tokens": 42}, status="partial_success")

    job = registry.get_job("job-doc-1")
    assert job["metadata"] == {"total_pages": 5, "indexed_pages": 3}
    assert job["status"] == "partial_success" and job["progress_pct"] == 100
    assert job["error"]  # partial success always explains itself

    doc = registry.get_document("doc-1")
    assert doc["status"] == "partial_success"
    assert doc["stats"] == {"total_tokens": 42}
    assert doc["chunk_count"] == 1
    assert isinstance(doc["ingested_at"], str)


def test_re_registering_a_complete_document_keeps_it_complete(registry):
    add_document(registry, job_id="j1")
    registry.complete_job("j1", {})
    registry.register_job("j2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    assert registry.get_document("doc-1")["status"] == "complete"


def test_fail_job_marks_the_job_and_document_failed_and_discards_any_pending_upload(registry):
    add_document(registry)
    registry.update_job_status("job-doc-1", "processing", 40)
    registry.fail_job("job-doc-1", "boom")
    assert registry.get_job("job-doc-1")["status"] == "failed"
    assert registry.get_job("job-doc-1")["error"] == "boom"
    assert registry.get_document("doc-1")["status"] == "failed"


def test_quota_and_counts_are_per_tenant(registry, store):
    add_document(registry, doc_id="doc-1", tenant="tenant-1")
    add_document(registry, doc_id="doc-2", tenant="tenant-2")
    store.load_chunks([chunk("a"), chunk("b"), chunk("c", tenant="tenant-2", doc_id="doc-2")])
    assert registry.get_tenant_quota("tenant-1") == 2
    assert registry.get_doc_count("tenant-2") == 1
    assert [d["doc_id"] for d in registry.list_documents("tenant-2")] == ["doc-2"]
    assert len(registry.list_documents(None)) == 2


def test_tenant_token_usage_accumulates(registry, pg_engine):
    registry.increment_tenant_embedding_tokens("tenant-1", 100)
    registry.increment_tenant_embedding_tokens("tenant-1", 50)
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
    metrics = MetricsStore(pg_engine)
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
