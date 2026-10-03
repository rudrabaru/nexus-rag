"""
Voyage reranker: rerank-3 over HTTP. 200M free tokens, but without a payment method the rerank
limit is 3 requests and 10K tokens a minute (measured 2026-10-02; a bucket separate from
embeddings), so requests are paced like embeddings. At that limit a pool of 8 candidates of
~600 tokens fits (3.9K tokens measured) and a pool of 20 (~10K+) was rejected: use a small
rerank_candidates. See docs/phases/phase5_retrieval.md.
"""
import asyncio
import time
from typing import List

import httpx

from src.config import Settings
from src.embedding.pacing import WINDOW_SECONDS, RateWindow
from src.embedding.providers import RETRYABLE_STATUS, estimate_tokens
from src.retrieving.models import RetrievalResult, RetrievedChunk
from src.retrieving.rerank_common import RerankError, rescored, result

VOYAGE_RERANK_PRICE_PER_MILLION = {"rerank-3": 0.05, "rerank-3-lite": 0.02, "rerank-2.5": 0.05}  # list price; the first 200M tokens are free
VOYAGE_ATTEMPTS = 3
VOYAGE_TIMEOUT_SECONDS = 45.0


class VoyageReranker:
    name = "voyage"

    def __init__(self, api_key: str, base_url: str, model: str, window: RateWindow):
        self.api_key = api_key
        self.endpoint = f"{base_url.rstrip('/')}/rerank"
        self.model = model
        self.window = window  # its own bucket: Voyage limits rerank separately from embeddings

    async def _wait_for_window(self, tokens: int) -> None:
        while (wait := self.window.reserve(tokens)) > 0:
            await asyncio.sleep(wait)

    async def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> RetrievalResult:
        start = time.time()
        if not candidates:
            return result(query, top_k, start, [])
        if not self.api_key:
            raise RerankError("voyage: VOYAGE_API_KEY is not set")

        # Voyage counts the documents' tokens plus the query once per document.
        estimate = sum(estimate_tokens(c.text) + estimate_tokens(query) for c in candidates)
        # A 429 means this minute's budget is spent: wait for one request slot of the configured rate.
        rate_limit_wait = WINDOW_SECONDS / self.window.requests_per_minute
        last_error = None
        async with httpx.AsyncClient(timeout=VOYAGE_TIMEOUT_SECONDS) as client:
            for attempt in range(VOYAGE_ATTEMPTS):
                await self._wait_for_window(estimate)
                try:
                    response = await client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": self.model, "query": query, "documents": [c.text for c in candidates], "top_k": top_k},
                    )
                except httpx.TransportError as e:
                    last_error, delay = e, 2.0 ** attempt
                else:
                    if response.status_code < 400:
                        body = response.json()
                        order = [(r["index"], r["relevance_score"]) for r in body["data"]]
                        cost = body.get("usage", {}).get("total_tokens", 0) * VOYAGE_RERANK_PRICE_PER_MILLION.get(self.model, 0.0) / 1_000_000
                        return result(query, top_k, start, rescored(candidates, order, top_k), cost)
                    last_error = RerankError(f"HTTP {response.status_code}: {response.text[:200]}")
                    if response.status_code not in RETRYABLE_STATUS:
                        break
                    delay = rate_limit_wait if response.status_code == 429 else 2.0 ** attempt
                if attempt < VOYAGE_ATTEMPTS - 1:
                    await asyncio.sleep(delay)
        raise RerankError(f"voyage: failed after {VOYAGE_ATTEMPTS} attempts: {last_error}")


_voyage_rerank_window = None


def voyage_rerank_window(settings: Settings) -> RateWindow:
    """One pacing window per process, shared by every Voyage reranker built (the limit is per API key)."""
    global _voyage_rerank_window
    if _voyage_rerank_window is None:
        _voyage_rerank_window = RateWindow(settings.voyage_rpm, settings.voyage_tpm)
    return _voyage_rerank_window
