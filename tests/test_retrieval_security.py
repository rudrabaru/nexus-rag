import pytest
from unittest.mock import MagicMock, patch
from src.retrieving.sparse import SparseRetriever
from src.retrieving.vector_store import QdrantManager
from src.registry.mixins.sparse_index import SparseIndexMixin

class DummySparseIndex(SparseIndexMixin):
    def __init__(self, mock_conn):
        self.mock_conn = mock_conn
    def _get_conn(self):
        return self.mock_conn

def test_vector_store_security_boundary_blocks_global_in_production():
    """Ensure production calls to QdrantManager.search with tenant_id='ALL' or None return [] when allow_global=False."""
    with patch("src.retrieving.vector_store.QdrantClient"), patch.dict("os.environ", {"QDRANT_URL": "http://mock", "QDRANT_API_KEY": "mock"}):
        store = QdrantManager()
        store.client = MagicMock()
        results = store.search(query_embedding=[0.1, 0.2], top_k=5, tenant_id=None, allow_global=False)
        assert results == []
        results = store.search(query_embedding=[0.1, 0.2], top_k=5, tenant_id="ALL", allow_global=False)
        assert results == []
        # Client query_points should never have been called!
        store.client.query_points.assert_not_called()

def test_sparse_index_security_boundary_blocks_global_in_production():
    """Ensure production calls (allow_global=False) with tenant_id='ALL' or None return empty results."""
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = MagicMock()
    dummy = DummySparseIndex(mock_conn)
    # When allow_global is False (default) and tenant_id is None
    results, fallback = dummy.search_fts5("test query", tenant_id=None, allow_global=False)
    assert results == []
    assert fallback is False

    # When allow_global is False and tenant_id is "ALL"
    results, fallback = dummy.search_fts5("test query", tenant_id="ALL", allow_global=False)
    assert results == []
    assert fallback is False

def test_sparse_index_allows_global_in_evaluation_mode():
    """Ensure evaluation calls (allow_global=True) execute SQL without tenant filter."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"chunk_id": "c1", "source_document": "doc1.md", "chunk_text": "hello world", "score": -5.0}
    ]
    mock_conn.__enter__.return_value.execute.return_value = mock_cursor

    dummy = DummySparseIndex(mock_conn)
    results, fallback = dummy.search_fts5("hello world", tenant_id="ALL", allow_global=True)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    # Verify the SQL executed did NOT contain AND tenant_id = ?
    call_args = mock_conn.__enter__.return_value.execute.call_args[0]
    sql_executed = call_args[0]
    assert "AND tenant_id = ?" not in sql_executed

def test_sparse_retriever_initialization_and_retrieve():
    """Test SparseRetriever Mode B wrapper."""
    import asyncio
    mock_registry = MagicMock()
    mock_registry.search_fts5.return_value = (
        [{"chunk_id": "c1", "source_document": "doc.md", "chunk_text": "sample text", "score": 4.5}],
        False
    )
    retriever = SparseRetriever(registry=mock_registry)
    res = asyncio.run(retriever.retrieve("sample", top_k=3, tenant_id="test_tenant", allow_global=False))
    assert res.query == "sample"
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "c1"
    assert res.embedding_latency_ms == 0.0
