"""Chunk store: vector, keyword and hybrid search, tenant isolation, the halfvec round trip."""
import numpy as np
import pytest
from sqlalchemy import select
from src.db.schema import EMBEDDING_DIMENSION, chunks, embedding_indexes
from src.retrieving.chunk_store import ChunkStore
from src.retrieving.chunk_writes import existing_source_urls
from src.retrieving.config import RetrievalConfig
from src.retrieving.dense import DenseRetriever
from src.retrieving.pipeline import RetrievalPipeline
from src.retrieving.sparse import SparseRetriever
from tests.support.postgres import AxisEmbedder, TEST_INDEX, add_document, chunk, unit_vector


pytestmark = pytest.mark.usefixtures("clean_tables")


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
