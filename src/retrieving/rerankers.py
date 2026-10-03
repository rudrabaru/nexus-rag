"""
Rerankers: reorder a first-stage candidate pool with a model that reads query and chunk together.

Each reranker is a small class with one async method, rerank(query, candidates, top_k), and
raises on failure. Degradation (keep the first-stage order and record why) is decided once, in
the pipeline (src/retrieving/pipeline.py), so an evaluation can see that a run was degraded
instead of silently measuring the first stage.

- flashrank (default): a small ONNX cross-encoder on CPU, inside the API process. No torch, no
  network after the one-time model download, $0 per query. Model choice (measured 2026-09-29,
  20 candidates of ~600 tokens, 16 CPU threads): ms-marco-TinyBERT-L-2-v2 ~95 ms,
  ms-marco-MiniLM-L-12-v2 ~1.95 s. TinyBERT is the default because the free API host has
  0.1 vCPU, where MiniLM would take on the order of 20 s per query; MiniLM is one setting
  away (FLASHRANK_MODEL) for an evaluation that wants to measure the trade-off. The hosted
  weights are CC-BY-SA and trained on MS MARCO (non-commercial terms).
- jina: jina-reranker-v2-base-multilingual over HTTP. Draws on Jina's one-time token grant.
- voyage: rerank-3 over HTTP. 200M free tokens, but without a payment method the rerank limit is
  3 requests and 10K tokens a minute (measured 2026-10-02; a bucket separate from embeddings), so
  requests are paced like embeddings. At that limit a pool of 8 candidates of ~600 tokens fits
  (3.9K tokens measured) and a pool of 20 (~10K+) was rejected: use a small rerank_candidates.

A local bge-reranker-v2-m3 is a future option; it is one more class.
"""
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import List

import httpx

from src.config import Settings
from src.retrieving.models import RetrievalResult, RetrievedChunk
from src.retrieving.rerank_common import RerankError, Reranker, rescored, result
from src.retrieving.voyage_reranker import VoyageReranker, voyage_rerank_window

logger = logging.getLogger(__name__)

FLASHRANK_MAX_TOKENS = 512  # BERT-family limit; chunks average ~570 tokens, so this is the most the model can read
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
JINA_COST_PER_1K_TOKENS = 0.000015  # list price, for cost reporting only
JINA_ATTEMPTS = 3
JINA_TIMEOUT_SECONDS = 45.0


class FlashRankReranker:
    name = "flashrank"

    def __init__(self, model: str, cache_dir: str):
        self.model = model
        self.cache_dir = cache_dir
        self._ranker = None
        self._lock = threading.Lock()

    def load(self):
        """Loads (and on first use downloads) the model. Called at API startup so no query pays for it."""
        with self._lock:
            if self._ranker is None:
                from flashrank import Ranker  # the API image only; the parse worker never reranks

                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                self._ranker = Ranker(model_name=self.model, cache_dir=self.cache_dir, max_length=FLASHRANK_MAX_TOKENS)
                logger.info(f"FlashRank model {self.model} loaded from {self.cache_dir}")
        return self._ranker

    def _order(self, query: str, candidates: List[RetrievedChunk]) -> List[tuple]:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": c.text} for i, c in enumerate(candidates)]
        ranked = self.load().rerank(RerankRequest(query=query, passages=passages))
        return [(p["id"], p["score"]) for p in ranked]

    async def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> RetrievalResult:
        start = time.time()
        if not candidates:
            return result(query, top_k, start, [])
        try:
            order = await asyncio.to_thread(self._order, query, candidates)  # CPU-bound: off the event loop
        except Exception as e:
            raise RerankError(f"flashrank: {type(e).__name__}: {e}") from e
        return result(query, top_k, start, rescored(candidates, order, top_k))


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

        last_error = None
        async with httpx.AsyncClient(timeout=JINA_TIMEOUT_SECONDS) as client:
            for attempt in range(JINA_ATTEMPTS):
                try:
                    response = await client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": JINA_RERANK_MODEL, "query": query, "documents": [c.text for c in candidates], "top_n": top_k},
                    )
                    response.raise_for_status()
                    body = response.json()
                    order = [(r["index"], r["relevance_score"]) for r in body["results"]]
                    tokens = body.get("usage", {}).get("total_tokens", 0)
                    return result(query, top_k, start, rescored(candidates, order, top_k), tokens / 1000 * JINA_COST_PER_1K_TOKENS)
                except Exception as e:
                    last_error = e
                    if attempt < JINA_ATTEMPTS - 1:
                        await asyncio.sleep(2 ** attempt)
        raise RerankError(f"jina: failed after {JINA_ATTEMPTS} attempts: {last_error}")


def default_flashrank_cache_dir() -> str:
    return str(Path.home() / ".cache" / "flashrank")


def build_reranker(name: str, settings: Settings) -> Reranker:
    if name == "flashrank":
        return FlashRankReranker(settings.flashrank_model, settings.flashrank_cache_dir or default_flashrank_cache_dir())
    if name == "jina":
        return JinaReranker(settings.jina_api_key)
    if name == "voyage":
        return VoyageReranker(settings.voyage_api_key, settings.voyage_base_url, settings.voyage_rerank_model, voyage_rerank_window(settings))
    raise ValueError(f"Unknown reranker {name!r}; expected flashrank, jina or voyage.")
