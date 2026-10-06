"""
FlashRank: a small ONNX cross-encoder on CPU, inside the API process. No torch, no network after the
one-time model download, $0 per query.

Model choice (measured 2026-09-29, 20 candidates of ~600 tokens, 16 CPU threads):
ms-marco-TinyBERT-L-2-v2 ~95 ms, ms-marco-MiniLM-L-12-v2 ~1.95 s. TinyBERT is the default because
the free API host has 0.1 vCPU, where MiniLM would take on the order of 20 s per query; MiniLM is one
setting away (FLASHRANK_MODEL) for an evaluation that wants to measure the trade-off. The hosted
weights are CC-BY-SA and trained on MS MARCO (non-commercial terms). Its scores are uncalibrated
(0.0013 for a correct chunk where MiniLM gave 0.93): they order results and are never a cutoff.
"""
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import List

from src.retrieving.models import RetrievalResult, RetrievedChunk
from src.retrieving.rerankers.base import RerankError, rescored, result

logger = logging.getLogger(__name__)

FLASHRANK_MAX_TOKENS = 512  # BERT-family limit; chunks average ~570 tokens, so this is the most the model can read


def default_flashrank_cache_dir() -> str:
    return str(Path.home() / ".cache" / "flashrank")


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
