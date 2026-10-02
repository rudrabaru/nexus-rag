from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_document: str
    source_url: Optional[str] = None
    text: str
    similarity_score: float
    metadata: Dict[str, Any]


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    embedding_tokens: int = 0  # spent by this search (0 when the query embedding was cached)
    embedding_cost_usd: float = 0.0
    # What embedding this query costs, cached or not: an evaluation compares configurations
    # on this, or every trial after the first would look cheaper for reusing the cache.
    query_embedding_tokens: int = 0
    rerank_cost_usd: float = 0.0
    chunks: List[RetrievedChunk]
    # The first-stage pool the reranker reordered (empty without a reranker), for rerank forensics.
    candidates: List[RetrievedChunk] = Field(default_factory=list)
    # Why this result is not what the configuration asked for, e.g. hybrid served sparse-only
    # because the query could not be embedded. An evaluation treats such a run as invalid.
    degraded: List[str] = Field(default_factory=list)
