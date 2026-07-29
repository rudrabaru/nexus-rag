import time
import logging
import os
import httpx
from typing import List

from src.retrieving.models import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

class OptionalReranker:
    def __init__(self):
        self.jina_api_key = os.environ.get("JINA_API_KEY")
        self.available = bool(self.jina_api_key)
        if not self.available:
            logger.warning("JINA_API_KEY not set. Reranker unavailable.")

    async def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> RetrievalResult:
        if not candidates:
            return RetrievalResult(query=query, top_k=top_k, latency_ms=0, chunks=[])

        start_time = time.time()

        if not self.available:
            logger.debug("Reranker not available, returning original ranking.")
            return RetrievalResult(
                query=query, 
                top_k=top_k, 
                latency_ms=(time.time() - start_time) * 1000, 
                chunks=candidates[:top_k]
            )

        chunks_text = [chunk.text for chunk in candidates]
        
        import asyncio
        rerank_tokens = 0
        async with httpx.AsyncClient(timeout=45.0) as client:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={"Authorization": f"Bearer {self.jina_api_key}"},
                        json={
                            "model": "jina-reranker-v2-base-multilingual",
                            "query": query,
                            "documents": chunks_text,
                            "top_n": top_k
                        }
                    )
                    response.raise_for_status()
                    resp_json = response.json()
                    results = resp_json["results"]
                    rerank_tokens = resp_json.get("usage", {}).get("total_tokens", 0)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Reranker API failed after {max_retries} attempts: {e}. Gracefully degrading to un-reranked chunks.")
                        return RetrievalResult(
                            query=query, 
                            top_k=top_k, 
                            latency_ms=(time.time() - start_time) * 1000, 
                            chunks=candidates[:top_k]
                        )
                    await asyncio.sleep(2 ** attempt)
            
        reranked_chunks = []
        for res in results:
            idx = res["index"]
            score = res["relevance_score"]
            chunk = candidates[idx]
            chunk.similarity_score = score
            reranked_chunks.append(chunk)

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            rerank_latency_ms=latency,
            rerank_tokens=rerank_tokens,
            chunks=reranked_chunks,
        )
