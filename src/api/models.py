from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = False
    evaluate_faithfulness: bool = False
    history: List[Dict[str, str]] = Field(default_factory=list)
    tenant_id: Optional[str] = None


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
