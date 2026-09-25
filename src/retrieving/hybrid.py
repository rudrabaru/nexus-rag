import asyncio
import time
from typing import Optional, Any

from src.retrieving.models import RetrievalResult
from src.retrieving.dense import DenseRetriever
from src.retrieving.sparse import SparseRetriever

RRF_K = 60  # the constant from Cormack et al. (2009); item 9 makes it a search knob


class HybridRetriever:
    """Dense + sparse retrieval fused by Reciprocal Rank Fusion on the shared chunk_id."""

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Any = None, allow_global: bool = False
    ) -> RetrievalResult:
        start_time = time.time()

        dense_result, sparse_result = await asyncio.gather(
            self.dense_retriever.retrieve(query, top_k=top_k, tenant_id=tenant_id, allow_global=allow_global),
            self.sparse_retriever.retrieve(
                query, top_k=top_k, tenant_id=tenant_id, pipeline_logger=pipeline_logger, allow_global=allow_global
            ),
        )

        scores = {}
        chunk_map = {}
        for ranked in (dense_result.chunks, sparse_result.chunks):
            for rank, chunk in enumerate(ranked):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                chunk_map.setdefault(chunk.chunk_id, chunk)

        sorted_chunk_ids = sorted(scores, key=scores.get, reverse=True)

        final_chunks = []
        max_score = scores[sorted_chunk_ids[0]] if sorted_chunk_ids else 1.0
        for cid in sorted_chunk_ids[:top_k]:
            chunk = chunk_map[cid]
            chunk.similarity_score = scores[cid] / max_score
            final_chunks.append(chunk)

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=(time.time() - start_time) * 1000,
            embedding_latency_ms=dense_result.embedding_latency_ms,
            search_latency_ms=max(dense_result.search_latency_ms, sparse_result.search_latency_ms),
            embedding_tokens=dense_result.embedding_tokens,
            chunks=final_chunks,
        )
