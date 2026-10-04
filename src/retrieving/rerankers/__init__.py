"""
Rerankers: reorder a first-stage candidate pool with a model that reads query and chunk together.

Each reranker is a small class with one async method, rerank(query, candidates, top_k), and raises
RerankError on failure. Degradation (keep the first-stage order and record why) is decided once, in
the pipeline (src/retrieving/pipeline.py), so an evaluation can see that a run was degraded instead
of silently measuring the first stage.

- flashrank (default): a small ONNX cross-encoder on CPU, inside the API process. $0 per query.
- jina: jina-reranker-v2-base-multilingual over HTTP; draws on Jina's one-time token grant.
- voyage: rerank-3 over HTTP; 200M free tokens, but 3 requests and 10K tokens a minute without a
  payment method, so its requests are paced and its candidate pool must be small.

A local bge-reranker-v2-m3 is a future option; it is one more module here.
"""
from src.config import Settings
from src.retrieving.rerankers.base import RerankError, Reranker
from src.retrieving.rerankers.flashrank import FlashRankReranker, default_flashrank_cache_dir
from src.retrieving.rerankers.jina import JinaReranker
from src.retrieving.rerankers.voyage import VoyageReranker, voyage_rerank_window

__all__ = ["FlashRankReranker", "JinaReranker", "RerankError", "Reranker", "VoyageReranker", "build_reranker"]


def build_reranker(name: str, settings: Settings) -> Reranker:
    if name == "flashrank":
        return FlashRankReranker(settings.flashrank_model, settings.flashrank_cache_dir or default_flashrank_cache_dir())
    if name == "jina":
        return JinaReranker(settings.jina_api_key.get_secret_value())
    if name == "voyage":
        return VoyageReranker(
            settings.voyage_api_key.get_secret_value(), settings.voyage_base_url, settings.voyage_rerank_model,
            voyage_rerank_window(settings),
        )
    raise ValueError(f"Unknown reranker {name!r}; expected flashrank, jina or voyage.")
