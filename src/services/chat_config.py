"""
The retrieval configuration chat runs for one request.

Layers, last wins: the environment's defaults, then what the workspace chose (the winner an
experiment promoted, src/stores/workspace.py), then the request's own top_k and reranker switch.
The result is a RetrievalConfig, run by the same pipeline an experiment trial runs, so what an
experiment measured is what chat serves.
"""
from typing import Any, Dict, Optional

from src.config import Settings
from src.retrieving.config import MAX_CANDIDATES, RetrievalConfig

# The reranker is given this many results per requested result to reorder, unless the workspace
# chose a pool size. Four balances candidate depth against rerank latency, which grows linearly with the pool. Experiment: not tuned on a corpus.
DEFAULT_RERANK_POOL_FACTOR = 4

# The fields a workspace may set. index_id and top_k are not among them: the index is the
# deployment's, and top_k is the caller's.
WORKSPACE_FIELDS = ("strategy", "rrf_k", "dense_weight", "sparse_weight", "reranker", "rerank_candidates")


def environment_defaults(settings: Settings) -> Dict[str, Any]:
    return {"strategy": settings.retrieval_strategy.lower(), "reranker": settings.effective_reranker}


def chat_retrieval_config(
    settings: Settings, workspace: Optional[Dict[str, Any]], top_k: int, use_reranker: bool
) -> RetrievalConfig:
    chosen = {**environment_defaults(settings), **{k: v for k, v in (workspace or {}).items() if k in WORKSPACE_FIELDS}}
    reranker = chosen.get("reranker") if use_reranker else None
    pool = chosen.get("rerank_candidates") or top_k * DEFAULT_RERANK_POOL_FACTOR
    return RetrievalConfig(
        **{**chosen, "top_k": top_k, "reranker": reranker, "rerank_candidates": min(max(pool, top_k), MAX_CANDIDATES)}
    )
