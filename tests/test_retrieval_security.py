from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.retrieving.chunk_store import ChunkStore, tenant_scope
from src.retrieving.chunk_writes import write_chunks
from src.retrieving.models import RetrievedChunk
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
        "chunk_id": chunk_id, "tenant_id": "tenant-1", "index_id": "voyage:voyage-4", "doc_id": "doc-1", "source_document": "Doc",
        "source_url": "https://a.example", "title": "Doc", "section_title": "S", "heading_path": ["H1", "H2"],
        "content_type": "text", "contains_code": False, "contains_table": False, "chunk_version": "v",
        "document_version": "v", "chunk_text": "hello world", "distance": 0.25, "score": 0.5,
    }
    row.update(overrides)
    return row


def make_store(rows=()):
    sync_engine = MagicMock()
    async_engine = RecordingAsyncEngine(rows)
    return ChunkStore(sync_engine, async_engine, "voyage:voyage-4"), sync_engine, async_engine


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


def test_tenant_scope_has_no_global_bypass():
    """There is no opt-in to an all-tenant search any more: evaluations are tenant-scoped too."""
    for tenant in (None, "", "ALL", "*"):
        assert tenant_scope(tenant) is None
    assert tenant_scope("tenant-1") == "tenant-1"


async def test_tenant_search_filters_on_the_tenant():
    store, _, async_engine = make_store([make_row()])
    await store.search_dense(EMBEDDING, top_k=5, tenant_id="tenant-1")
    assert "chunks.tenant_id =" in where_clause(async_engine.statements[0])


async def test_both_searches_are_scoped_to_the_stores_index():
    """Vectors of different models are not comparable, and sparse must search the same rows as dense."""
    store, _, async_engine = make_store([make_row()])
    await store.search_dense(EMBEDDING, top_k=5, tenant_id="tenant-1")
    await store.search_sparse("hello", tenant_id="tenant-1")
    for stmt in async_engine.statements:
        assert "chunks.index_id =" in where_clause(stmt)
        assert stmt.compile().params["index_id_1"] == "voyage:voyage-4"


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


async def test_heading_path_is_a_typed_list_on_the_chunk():
    """context_builder and evaluation_helpers json.loads() this field; a list would break heading matching."""
    store, _, _ = make_store([make_row(heading_path=["H1", "H2"])])
    [chunk] = await store.search_dense(EMBEDDING, top_k=5, tenant_id="tenant-1")
    assert chunk.heading_path == ["H1", "H2"] and "heading_path" not in chunk.metadata


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
        chunk_id="c", source_url="u", source_document="d", title="t", chunk_text="x",
        token_count=1, document_version="v", chunk_version="v",
        tenant_id="tenant-1", doc_id="doc-1", embedding=EMBEDDING, embedding_model="m", index_id="test:m",
    )
    fields[missing] = None
    conn = MagicMock()
    with pytest.raises(ValueError, match="refusing to store"):
        write_chunks(conn, [EmbeddedChunk(**fields)])
    conn.execute.assert_not_called()
