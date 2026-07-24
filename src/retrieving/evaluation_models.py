from pydantic import BaseModel
from typing import List, Dict, Any

class EvaluationQuery(BaseModel):
    query: str
    expected_topic: str
    expected_content_type: str = "concept"
    acceptable_documents: List[str]
    acceptable_headings: List[str]
    difficulty: str
    category: str


class RetrievedChunkInfo(BaseModel):
    chunk_id: str
    source_document: str
    similarity_score: float
    text: str
    section_title: str = ""
    heading_path: List[str] = []
    metadata: Dict[str, Any] = {}

    raw_identifier: str = ""
    normalized_identifier: List[str] = []
    matched_target: str = ""
    matching_rule_used: str = ""
    classification_reason: str = ""
    match_type: str = "No Match"


class EvaluationResult(BaseModel):
    query: str
    expected_topic: str
    difficulty: str
    category: str
    retrieved_documents: List[str]
    retrieved_chunks: List[RetrievedChunkInfo]
    rank: int = -1
    exact_match_rank: int = -1
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    best_match_type: str = "No Match"
    latency_ms: float
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0


class EvaluationReport(BaseModel):
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float
    results: List[EvaluationResult]
