from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = False
    evaluate_faithfulness: bool = False
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)


class SourceDocument(BaseModel):
    url: str
    section: str
    similarity_score: float
    chunk_preview: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    faithfulness_score: Optional[float] = None
    faithfulness_reasoning: Optional[str] = None
    latency_ms: float
    latency_breakdown: Optional[Dict[str, float]] = None
