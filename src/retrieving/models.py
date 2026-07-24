from typing import List, Dict, Any, Optional
from pydantic import BaseModel

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
    embedding_tokens: int = 0
    rerank_tokens: int = 0
    chunks: List[RetrievedChunk]
