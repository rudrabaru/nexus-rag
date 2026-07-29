import time
from typing import Optional, Any
from src.registry.database import DocumentRegistry
from src.retrieving.models import RetrievedChunk, RetrievalResult

class SparseRetriever:
    """Retrieves document chunks using SQLite FTS5 sparse keyword search."""

    def __init__(self, registry: Optional[DocumentRegistry] = None):
        self.registry = registry or DocumentRegistry()

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Optional[Any] = None, allow_global: bool = False
    ) -> RetrievalResult:
        start_time = time.time()
        fts_results, fallback_used = self.registry.search_fts5(
            query=query, tenant_id=tenant_id, limit=top_k, allow_global=allow_global
        )
        if fallback_used and pipeline_logger:
            pipeline_logger.log_event("fts_fallback_triggered", query=query, tenant_id=tenant_id)

        chunks = []
        for res in fts_results:
            chunks.append(
                RetrievedChunk(
                    chunk_id=res["chunk_id"],
                    source_document=res["source_document"],
                    source_url=res.get("source_url"),
                    text=res["chunk_text"],
                    similarity_score=res["score"],
                    metadata=res,
                )
            )

        latency = (time.time() - start_time) * 1000
        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=0.0,
            search_latency_ms=latency,
            chunks=chunks,
        )
