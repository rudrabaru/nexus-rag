"""What every reranker shares: its failure type, its interface, and building a RetrievalResult from a reordering."""
import time
from typing import List, Protocol

from src.retrieving.models import RetrievalResult, RetrievedChunk


class RerankError(RuntimeError):
    """The reranker could not score the candidates."""


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> RetrievalResult: ...


def result(query: str, top_k: int, start: float, chunks: List[RetrievedChunk], cost_usd: float = 0.0) -> RetrievalResult:
    latency = (time.time() - start) * 1000
    return RetrievalResult(
        query=query, top_k=top_k, latency_ms=latency, rerank_latency_ms=latency, rerank_cost_usd=cost_usd, chunks=chunks
    )


def rescored(candidates: List[RetrievedChunk], order: List[tuple], top_k: int) -> List[RetrievedChunk]:
    """Copies of the candidates in reranked order, carrying the reranker's score. order: (index, score)."""
    return [candidates[i].model_copy(update={"similarity_score": float(score)}) for i, score in order[:top_k]]
