from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.retrieving.config import MAX_CANDIDATES, RerankerName, RetrievalConfig, Strategy


class WorkspaceStats(BaseModel):
    documents_count: int
    total_chunks: int
    total_tokens: int


class WorkspaceRetrieval(BaseModel):
    """The retrieval settings a workspace's chat runs. Anything left out keeps the deployment's default."""

    strategy: Optional[Strategy] = None
    rrf_k: Optional[int] = Field(default=None, ge=1)
    dense_weight: Optional[float] = Field(default=None, ge=0.0)
    sparse_weight: Optional[float] = Field(default=None, ge=0.0)
    reranker: Optional[RerankerName] = None
    rerank_candidates: Optional[int] = Field(default=None, ge=1, le=MAX_CANDIDATES)

    @model_validator(mode="after")
    def runs_as_a_configuration(self):
        RetrievalConfig(**self.chosen())  # raises when the combination is not a valid configuration
        return self

    def chosen(self) -> dict:
        return self.model_dump(exclude_none=True)


class WorkspaceSettingsResponse(BaseModel):
    source: Literal["workspace", "default"]
    retrieval: WorkspaceRetrieval


class UsageEntry(BaseModel):
    log_id: int
    timestamp: str
    query: str
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    faithfulness_score: Optional[float] = None
    provider: Optional[str] = None
    total_cost_usd: Optional[float] = None


class UsageSummary(BaseModel):
    total_queries: int
    total_cost_usd: float
    avg_cost_per_query_usd: float
    avg_latency_ms: float


class UsageResponse(BaseModel):
    summary: UsageSummary
    queries: List[UsageEntry]
