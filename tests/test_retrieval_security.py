import json
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.retrieving.chunk_store import ChunkStore, resolve_tenant_scope
from src.retrieving.hybrid import HybridRetriever
from src.retrieving.models import RetrievalResult, RetrievedChunk
from src.retrieving.sparse import SparseRetriever

EMBEDDING = [0.1] * 1024


class RecordingAsyncEngine:
    """Stands in for the async engine: records every statement and returns canned rows."""

    def __init__(self, rows=()):
        self.statements = []
        self.rows = list(rows)

    def connect(self):
        engine = self

        class _Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, stmt):
                engine.statements.append(stmt)
                result = MagicMock()
                result.mappings.return_value.all.return_value = engine.rows
                return result

        return _Conn()


def sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def where_clause(stmt) -> str:
    return sql(stmt).split("WHERE", 1)[1] if "WHERE" in sql(stmt) else ""


def make_row(chunk_id="md5_chunk_000", **overrides):
    row = {
        "chunk_id": chunk_id, "tenant_id": "tenant-1", "doc_id": "doc-1", "source_document": "Doc",
        "source_url": "https://a.example", "title": "Doc", "section_title": "S", "heading_path": ["H1", "H2"],
        "content_type": "text", "contains_code": False, "contains_table": False, "chunk_version": "v",
        "document_version": "v", "chunk_text": "hello world", "distance": 0.25, "score": 0.5,
    }
    row.update(overrides)
    return row


def make_store(rows=()):
    sync_engine = MagicMock()
    async_engine = RecordingAsyncEngine(rows)
    return ChunkStore(sync_engine, async_engine), sync_engine, async_engine


# ── Default-deny: no tenant means no results and no SQL ─────────────────────

@pytest.mark.parametrize("tenant", [None, "", "ALL", "*"])
async def test_dense_search_without_a_tenant_returns_nothing_and_runs_no_sql(tenant):
    store, sync_engine, async_engine = make_store([make_row()])
    assert await store.search_dense(EMBEDDING, top_k=5, tenant_id=tenant) == []
    assert async_engine.statements == []
    sync_engine.connect.assert_not_called()


@pytest.mark.parametrize("tenant", [None, "", "ALL", "*"])
async def test_sparse_search_without_a_tenant_returns_nothing_and_runs_no_sql(tenant):
    store, _, async_engine = make_store([make_row()])
    assert await store.search_sparse("hello", tenant_id=tenant) == ([], False)
    assert async_engine.statements == []


def test_tenant_scope_resolution():
    assert resolve_tenant_scope(None, False) == (False, None)
    assert resolve_tenant_scope("ALL", False) == (False, None)
    assert resolve_tenant_scope("ALL", True) == (True, None)
    assert resolve_tenant_scope("tenant-1", False) == (True, "tenant-1")
    assert resolve_tenant_scope("tenant-1", True) == (True, "tenant-1")


async def test_tenant_search_filters_on_the_tenant():
    store, _, async_engine = make_store([make_row()])
    await store.search_dense(EMBEDDING, top_k=5, tenant_id="tenant-1")
    assert "chunks.tenant_id =" in where_clause(async_engine.statements[0])


async def test_global_search_is_opt_in_and_unfiltered():
    store, _, async_engine = make_store([make_row()])
    await store.search_dense(EMBEDDING, top_k=5, tenant_id=None, allow_global=True)
    assert "tenant_id" not in where_clause(async_engine.statements[0])


async def test_dense_search_orders_by_cosine_distance_and_limits():
    store, _, async_engine = make_store([make_row()])
    await store.search_dense(EMBEDDING, top_k=7, tenant_id="tenant-1")
    compiled = sql(async_engine.statements[0])
    assert "<=>" in compiled and "ORDER BY" in compiled and "LIMIT" in compiled


# ── Result shape ─────────────────────────────────────────────────────────────

async def test_results_carry_the_real_chunk_id_and_cosine_similarity():
    store, _, _ = make_store([make_row(chunk_id="md5_chunk_007", distance=0.25)])
    [chunk] = await store.search_dense(EMBEDDING, top_k=5, tenant_id="tenant-1")
    assert chunk.chunk_id == "md5_chunk_007"
    assert chunk.similarity_score == pytest.approx(0.75)
    assert chunk.text == "hello world"


async def test_heading_path_keeps_the_json_string_shape_consumers_parse():
    """context_builder and evaluation_helpers json.loads() this field; a list would break heading matching."""
    store, _, _ = make_store([make_row(heading_path=["H1", "H2"])])
    [chunk] = await store.search_dense(EMBEDDING, top_k=5, tenant_id="tenant-1")
    assert json.loads(chunk.metadata["heading_path"]) == ["H1", "H2"]


# ── Sparse search ────────────────────────────────────────────────────────────

async def test_sparse_falls_back_from_all_terms_to_any_term():
    store, _, async_engine = make_store(rows=[])
    chunks, fallback = await store.search_sparse("alpha beta", tenant_id="tenant-1")
    assert chunks == [] and fallback is True
    assert len(async_engine.statements) == 2
    assert "plainto_tsquery" in sql(async_engine.statements[0])
    assert "replace" in sql(async_engine.statements[1])


async def test_sparse_hit_on_all_terms_skips_the_fallback():
    store, _, async_engine = make_store([make_row()])
    chunks, fallback = await store.search_sparse("hello world", tenant_id="tenant-1")
    assert len(chunks) == 1 and fallback is False
    assert len(async_engine.statements) == 1


@pytest.mark.parametrize("query", ["", "   "])
async def test_blank_sparse_query_runs_no_sql(query):
    store, _, async_engine = make_store([make_row()])
    assert await store.search_sparse(query, tenant_id="tenant-1") == ([], False)
    assert async_engine.statements == []


async def test_sparse_retriever_wraps_the_store():
    store = MagicMock()

    async def search_sparse(*args, **kwargs):
        return [RetrievedChunk(chunk_id="c1", source_document="d", text="t", similarity_score=1.0, metadata={})], False

    store.search_sparse = search_sparse
    result = await SparseRetriever(store).retrieve("sample", top_k=3, tenant_id="tenant-1")
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.embedding_latency_ms == 0.0


# ── Writes ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("missing", ["tenant_id", "doc_id"])
def test_a_chunk_without_tenant_or_document_is_never_stored(missing):
    from src.embedding.models import EmbeddedChunk

    fields = dict(
        chunk_id="c", source_url="u", source_document="d", title="t", chunk_index=0, chunk_text="x",
        token_count=1, char_start=0, char_end=1, document_version="v", chunk_version="v",
        tenant_id="tenant-1", doc_id="doc-1", embedding=EMBEDDING, embedding_model="m",
    )
    fields[missing] = None
    store, sync_engine, _ = make_store()
    with pytest.raises(ValueError):
        store.load_chunks([EmbeddedChunk(**fields)])
    sync_engine.begin.assert_not_called()


# ── Hybrid fusion ────────────────────────────────────────────────────────────

class FixedRetriever:
    def __init__(self, chunk_ids):
        self.chunk_ids = chunk_ids

    async def retrieve(self, query, top_k=5, tenant_id=None, pipeline_logger=None, allow_global=False):
        chunks = [
            RetrievedChunk(chunk_id=cid, source_document="d", text=cid, similarity_score=1.0, metadata={})
            for cid in self.chunk_ids
        ]
        return RetrievalResult(query=query, top_k=top_k, latency_ms=1.0, chunks=chunks)


async def test_a_chunk_found_by_both_retrievers_is_fused_not_duplicated():
    """
    Regression: dense results used Qdrant's UUID point id while sparse results used the real
    chunk_id, so RRF never matched them and one chunk could fill two result slots.
    """
    hybrid = HybridRetriever(FixedRetriever(["shared", "dense-only"]), FixedRetriever(["sparse-only", "shared"]))
    result = await hybrid.retrieve("q", top_k=5, tenant_id="tenant-1")

    ids = [c.chunk_id for c in result.chunks]
    assert sorted(ids) == ["dense-only", "shared", "sparse-only"]
    assert ids[0] == "shared"  # ranks 1 + 2 outscore any single-list rank
