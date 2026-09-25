import time
import logging
import os
import httpx
from typing import Optional, Any

from src.retrieving.chunk_store import ChunkStore
from src.embedding.config import EmbeddingConfig
from src.retrieving.models import RetrievalResult

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(
        self,
        chunk_store: ChunkStore,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.chunk_store = chunk_store
        self.config = embedding_config or EmbeddingConfig()
        
        logger.info(
            f"Loading embedding model for dense retrieval: {self.config.model_name}"
        )
        self.jina_api_key = os.environ.get("JINA_API_KEY")

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Optional[Any] = None, allow_global: bool = False
    ) -> RetrievalResult:
        start_time = time.time()

        embed_start = time.time()
        
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY not set for DenseRetriever.")
            
        import hashlib
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        
        if not hasattr(self, "_embedding_cache"):
            self._embedding_cache = {}
            self._MAX_CACHE_SIZE = 500

        if cache_key in self._embedding_cache:
            query_embedding = self._embedding_cache[cache_key]
            embedding_tokens = 0
            logger.info("Using cached embedding for query.")
        else:
            import asyncio
            embedding_tokens = 0
            async with httpx.AsyncClient(timeout=10.0) as client:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            "https://api.jina.ai/v1/embeddings",
                            headers={"Authorization": f"Bearer {self.jina_api_key}"},
                            json={
                                "model": self.config.model_name,
                                "input": [query],
                                "task": "retrieval.query"
                            }
                        )
                        response.raise_for_status()
                        resp_json = response.json()
                        query_embedding = resp_json["data"][0]["embedding"]
                        embedding_tokens = resp_json.get("usage", {}).get("total_tokens", 0)
                        
                        # Add to cache with FIFO eviction
                        if len(self._embedding_cache) >= self._MAX_CACHE_SIZE:
                            self._embedding_cache.pop(next(iter(self._embedding_cache)))
                        self._embedding_cache[cache_key] = query_embedding
                        
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        await asyncio.sleep(2 ** attempt)
            
        embed_latency = (time.time() - embed_start) * 1000

        search_start = time.time()
        candidates = await self.chunk_store.search_dense(
            query_embedding=query_embedding,
            top_k=top_k,
            tenant_id=tenant_id,
            allow_global=allow_global,
        )
        search_latency = (time.time() - search_start) * 1000
        latency = (time.time() - start_time) * 1000

        if candidates:
            scores_str = ", ".join(f"{c.similarity_score:.4f}" for c in candidates[:5])
            logger.info(
                f"Retrieved {len(candidates)} chunks for tenant={tenant_id}"
                f" | metric={self.chunk_store.distance_metric}"
                f" | top-5 scores: [{scores_str}]"
            )

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=embed_latency,
            search_latency_ms=search_latency,
            embedding_tokens=embedding_tokens,
            chunks=candidates,
        )
