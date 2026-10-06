import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

from src.embedding.embedder import Embedder
from src.retrieving.chunk_store import ChunkStore
from src.retrieving.models import RetrievalResult
from src.stores.query_embeddings import QueryEmbeddingStore, query_key

logger = logging.getLogger(__name__)

QUERY_CACHE_SIZE = 500  # repeated queries (UI retries, evaluation reruns) skip a paced API call


class DenseRetriever:
    def __init__(self, chunk_store: ChunkStore, embedder: Embedder, saved_queries: Optional[QueryEmbeddingStore] = None):
        if embedder.index_id != chunk_store.index_id:
            raise ValueError(
                f"Query embedder {embedder.index_id!r} does not match the index {chunk_store.index_id!r}; "
                "vectors of different models are not comparable."
            )
        self.chunk_store = chunk_store
        self.embedder = embedder
        self._cache: "OrderedDict[str, Tuple[List[float], int]]" = OrderedDict()
        self._saved = saved_queries  # set for experiments only: a chat question is never stored

    def _remember(self, key: str, vector: List[float], tokens: int) -> None:
        self._cache[key] = (vector, tokens)
        if len(self._cache) > QUERY_CACHE_SIZE:
            self._cache.popitem(last=False)

    async def _embed_query(self, query: str) -> Tuple[List[float], int, bool]:
        """(vector, tokens the query costs to embed, whether it came from a cache and so cost nothing now)."""
        key = query_key(query)
        if key in self._cache:
            vector, tokens = self._cache[key]
            return vector, tokens, True
        index_id = self.embedder.index_id
        if self._saved is not None:
            saved = await asyncio.to_thread(self._saved.get, index_id, key)
            if saved:
                self._remember(key, *saved)
                return saved[0], saved[1], True
        batch = await self.embedder.aembed([query], "query")
        self._remember(key, batch.vectors[0], batch.tokens)
        if self._saved is not None:
            try:
                await asyncio.to_thread(self._saved.put, index_id, key, batch.vectors[0], batch.tokens)
            except Exception:
                logger.exception("Could not keep the query embedding; the next run will embed it again")
        return batch.vectors[0], batch.tokens, False

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Optional[Any] = None
    ) -> RetrievalResult:
        start_time = time.time()
        query_embedding, query_tokens, cached = await self._embed_query(query)
        embedding_tokens = 0 if cached else query_tokens
        embed_latency = (time.time() - start_time) * 1000

        search_start = time.time()
        candidates = await self.chunk_store.search_dense(
            query_embedding=query_embedding, top_k=top_k, tenant_id=tenant_id
        )
        search_latency = (time.time() - search_start) * 1000

        if candidates:
            scores = ", ".join(f"{c.similarity_score:.4f}" for c in candidates[:5])
            logger.info(
                f"Retrieved {len(candidates)} chunks for tenant={tenant_id} | index={self.chunk_store.index_id}"
                f" | metric={self.chunk_store.distance_metric} | top-5 scores: [{scores}]"
            )

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=(time.time() - start_time) * 1000,
            embedding_latency_ms=embed_latency,
            search_latency_ms=search_latency,
            embedding_tokens=embedding_tokens,
            embedding_cost_usd=self.embedder.cost_usd(embedding_tokens),
            query_embedding_tokens=query_tokens,
            chunks=candidates,
        )
