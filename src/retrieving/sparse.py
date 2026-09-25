import time
from typing import Optional, Any

from src.retrieving.chunk_store import ChunkStore
from src.retrieving.models import RetrievalResult


class SparseRetriever:
    """Retrieves chunks by Postgres full-text search over the chunk text."""

    def __init__(self, chunk_store: ChunkStore):
        self.chunk_store = chunk_store

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Optional[Any] = None, allow_global: bool = False
    ) -> RetrievalResult:
        start_time = time.time()
        chunks, fallback_used = await self.chunk_store.search_sparse(
            query, tenant_id=tenant_id, limit=top_k, allow_global=allow_global
        )
        if fallback_used and pipeline_logger:
            pipeline_logger.log_event("fts_fallback_triggered", query=query, tenant_id=tenant_id)

        latency = (time.time() - start_time) * 1000
        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=0.0,
            search_latency_ms=latency,
            chunks=chunks,
        )
