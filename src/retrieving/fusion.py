"""
Weighted Reciprocal Rank Fusion.

    score(chunk) = sum over rankings i of  w_i / (k + rank_i(chunk)),   rank starting at 1

A chunk absent from a ranking contributes nothing from it. Fusion works on ranks, not raw
scores, because cosine similarities and ts_rank_cd scores live on unrelated scales. Kept as
our own ~20 lines rather than a library call because k and the weights are search knobs;
tests/test_retrieval_pipeline.py checks the unweighted case against ranx's implementation.
"""
from typing import Dict, List, Sequence, Tuple

from src.retrieving.models import RetrievedChunk


def rrf_scores(rankings: Sequence[Tuple[Sequence[str], float]], k: int) -> Dict[str, float]:
    """Fused score per chunk id, from (ranked chunk ids, weight) pairs."""
    scores: Dict[str, float] = {}
    for ids, weight in rankings:
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return scores


def fuse(rankings: Sequence[Tuple[Sequence[RetrievedChunk], float]], k: int, limit: int) -> List[RetrievedChunk]:
    """
    The `limit` best chunks by fused score. Returned chunks are copies whose similarity_score is
    the fused score scaled so the best is 1.0 (the retrievers' own scores are not comparable).
    Ties keep first-seen order, which favours the earlier ranking (dense before sparse).
    """
    rankings = [(chunks, weight) for chunks, weight in rankings if weight > 0]
    first_seen: Dict[str, RetrievedChunk] = {}
    for chunks, _ in rankings:
        for chunk in chunks:
            first_seen.setdefault(chunk.chunk_id, chunk)
    scores = rrf_scores([([c.chunk_id for c in chunks], weight) for chunks, weight in rankings], k)
    ordered = sorted(first_seen, key=lambda cid: scores[cid], reverse=True)[:limit]
    best = scores[ordered[0]] if ordered and scores[ordered[0]] > 0 else 1.0
    return [first_seen[cid].model_copy(update={"similarity_score": scores[cid] / best}) for cid in ordered]
