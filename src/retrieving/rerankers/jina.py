"""Jina reranker over HTTP. Draws on Jina's one-time token grant, so it is an option, not the default."""
import time
from typing import List

import httpx

from src.embedding.providers import RETRYABLE_STATUS
from src.retrieving.models import RetrievalResult, RetrievedChunk
from src.retrieving.rerankers.base import RerankError, rescored, result
from src.retry import RetryableError, retry_async

JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
JINA_COST_PER_1K_TOKENS = 0.000015  # list price, for cost reporting only
JINA_ATTEMPTS = 3
JINA_TIMEOUT_SECONDS = 45.0


class JinaReranker:
    name = "jina"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> RetrievalResult:
        start = time.time()
        if not candidates:
            return result(query, top_k, start, [])
        if not self.api_key:
            raise RerankError("jina: JINA_API_KEY is not set")

        async with httpx.AsyncClient(timeout=JINA_TIMEOUT_SECONDS) as client:
            async def send():
                try:
                    response = await client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": JINA_RERANK_MODEL, "query": query, "documents": [c.text for c in candidates], "top_n": top_k},
                    )
                except httpx.TransportError as e:
                    raise RetryableError(RerankError(f"jina: unreachable: {e}"))
                if response.status_code in RETRYABLE_STATUS:
                    raise RetryableError(RerankError(f"jina: HTTP {response.status_code}: {response.text[:200]}"))
                if response.status_code >= 400:  # a bad key or request fails the same way again
                    raise RerankError(f"jina: HTTP {response.status_code}: {response.text[:200]}")
                return response.json()

            body = await retry_async(send, JINA_ATTEMPTS, "RERANK jina")

        order = [(r["index"], r["relevance_score"]) for r in body["results"]]
        tokens = body.get("usage", {}).get("total_tokens", 0)
        return result(query, top_k, start, rescored(candidates, order, top_k), tokens / 1000 * JINA_COST_PER_1K_TOKENS)
