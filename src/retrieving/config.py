"""
One retrieval configuration: every query-time retrieval knob the evaluation platform varies.

Chat runs the default configuration (settings plus the request's top_k and reranker toggle);
an evaluation runs one configuration per trial. Both go through the same pipeline
(src/retrieving/pipeline.py), so a configuration measured offline is exactly what chat serves.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["dense", "sparse", "hybrid"]
RerankerName = Literal["flashrank", "jina"]

# hnsw.ef_search is 100 (src/registry/engine.py) and must be at least the largest LIMIT a
# dense search issues, so the candidate depth is capped below it.
MAX_CANDIDATES = 80


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index_id: Optional[str] = None  # provider:model; None = the configured embedding index
    strategy: Strategy = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    # Reciprocal Rank Fusion constant. 60 is Cormack et al. (2009)'s value, not a tuned one;
    # it is a knob precisely so an evaluation can measure it.
    rrf_k: int = Field(60, ge=1)
    dense_weight: float = Field(1.0, ge=0.0)
    sparse_weight: float = Field(1.0, ge=0.0)
    reranker: Optional[RerankerName] = None
    # How many first-stage candidates the reranker reorders. Chat uses top_k * 4 (the depth
    # the reranker has always seen); deeper pools cost rerank latency linearly.
    rerank_candidates: int = Field(20, ge=1, le=MAX_CANDIDATES)

    @model_validator(mode="after")
    def consistent(self):
        if self.reranker and self.rerank_candidates < self.top_k:
            raise ValueError("rerank_candidates must be at least top_k.")
        if self.strategy == "hybrid" and self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("A hybrid configuration needs a non-zero dense or sparse weight.")
        return self

    @property
    def candidate_count(self) -> int:
        """How many chunks the first stage returns: the rerank pool, or the final top_k."""
        return self.rerank_candidates if self.reranker else self.top_k
