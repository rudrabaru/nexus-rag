"""What a chat request carries through the service and what comes back: plain data, no behaviour."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.generating.models import GenerationResult
from src.retrieving.config import RetrievalConfig
from src.retrieving.models import RetrievalResult

EMPTY_WORKSPACE_MESSAGE = "This workspace has no documents yet. Add a document or a web page first."
GENERATION_FAILED_MESSAGE = "The language model could not produce an answer right now. Please try again shortly."
INTERNAL_ERROR_MESSAGE = "Something went wrong while answering."
SOURCE_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class ChatQuery:
    query: str
    top_k: int = 5
    use_reranker: bool = False
    evaluate_faithfulness: bool = False
    history: Tuple[Dict[str, str], ...] = ()


@dataclass
class SourceView:
    url: str
    section: str
    similarity_score: float
    chunk_preview: str


@dataclass
class Prepared:
    tenant_id: str
    chat: ChatQuery
    started: float
    empty: bool = False
    config: Optional[RetrievalConfig] = None
    retrieval: Optional[RetrievalResult] = None


@dataclass
class ChatAnswer:
    answer: str
    sources: List[SourceView] = field(default_factory=list)
    latency_ms: float = 0.0
    latency_breakdown: Optional[Dict[str, float]] = None
    result: Optional[GenerationResult] = None  # kept for the faithfulness check that may follow
    log_id: Optional[int] = None


@dataclass
class Comparison:
    baseline: List[SourceView]
    reranked: List[SourceView]
    baseline_latency_ms: float
    reranked_latency_ms: float
    reranker: Optional[str]
    degraded: List[str]


def preview(chunk) -> SourceView:
    section = " > ".join(chunk.heading_path)
    label = f"{chunk.source_document} > {section}" if chunk.source_document and section else chunk.source_document or section
    text = chunk.text
    return SourceView(chunk.source_url or "", label, chunk.similarity_score, text[:300] + ("..." if len(text) > 300 else ""))


def sources_of(result: GenerationResult) -> List[SourceView]:
    return [
        SourceView(
            url=chunk.source_url or "",
            section=" > ".join(chunk.heading_path) if chunk.heading_path else "",
            similarity_score=chunk.similarity_score,
            chunk_preview=chunk.text[:SOURCE_PREVIEW_CHARS] + ("..." if len(chunk.text) > SOURCE_PREVIEW_CHARS else ""),
        )
        for chunk in result.context_window.included_chunks
    ]
