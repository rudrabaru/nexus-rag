import time
from typing import Optional, Any

from src.retrieving.models import RetrievedChunk, RetrievalResult
from src.retrieving.dense import DenseRetriever

class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        registry: Any,
    ):
        self.dense_retriever = dense_retriever
        self.registry = registry

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Any = None, allow_global: bool = False
    ) -> RetrievalResult:
        start_time = time.time()

        dense_result = await self.dense_retriever.retrieve(
            query, top_k=top_k, tenant_id=tenant_id, allow_global=allow_global
        )
        dense_chunks = dense_result.chunks

        sparse_start = time.time()
        
        fts_results, fallback_used = self.registry.search_fts5(query, tenant_id, limit=top_k, allow_global=allow_global)
        if fallback_used and pipeline_logger:
            pipeline_logger.log_event("fts_fallback_triggered", query=query, tenant_id=tenant_id)
            
        sparse_chunks = []
        for res in fts_results:
            sparse_chunks.append(
                RetrievedChunk(
                    chunk_id=res["chunk_id"],
                    source_document=res["source_document"],
                    source_url=res.get("source_url"),
                    text=res["chunk_text"],
                    similarity_score=res["score"],
                    metadata=res,
                )
            )

        sparse_latency = (time.time() - sparse_start) * 1000

        rrf_k = 60
        scores = {}
        chunk_map = {}

        for rank, chunk in enumerate(dense_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        sorted_chunk_ids = sorted(
            scores.keys(), key=lambda cid: scores[cid], reverse=True
        )

        final_chunks = []
        max_score = scores[sorted_chunk_ids[0]] if sorted_chunk_ids else 1.0
        for cid in sorted_chunk_ids[:top_k]:
            c = chunk_map[cid]
            c.similarity_score = scores[cid] / max_score
            final_chunks.append(c)

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=dense_result.embedding_latency_ms,
            search_latency_ms=dense_result.search_latency_ms + sparse_latency,
            embedding_tokens=dense_result.embedding_tokens,
            chunks=final_chunks,
        )
