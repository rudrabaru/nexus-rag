from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.services.chat_models import ChatQuery

MAX_CHAT_TURNS = 20
MAX_TURN_CHARS = 4000  # a long answer pasted back; bounds the prompt a caller can make us pay for


class ChatTurn(BaseModel):
    """One earlier message. Only user and assistant turns exist: a caller cannot inject a "system" turn."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TURN_CHARS)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20, description="How many sources to answer from.")
    use_reranker: bool = Field(default=False, description="Reorder the candidates with the workspace's reranker.")
    evaluate_faithfulness: bool = Field(default=False, description="Have a judge model score the answer afterwards (costs LLM calls).")
    history: List[ChatTurn] = Field(default_factory=list, max_length=MAX_CHAT_TURNS)

    def to_query(self) -> ChatQuery:
        return ChatQuery(
            query=self.query, top_k=self.top_k, use_reranker=self.use_reranker,
            evaluate_faithfulness=self.evaluate_faithfulness,
            history=tuple(turn.model_dump() for turn in self.history),
        )


class Source(BaseModel):
    url: str
    section: str
    similarity_score: float
    chunk_preview: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: float
    latency_breakdown: Optional[Dict[str, float]] = None


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    content: List[Source]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    latency_ms: Optional[float] = None


class FaithfulnessContent(BaseModel):
    score: Optional[float] = None
    reasoning: Optional[str] = None


class FaithfulnessEvent(BaseModel):
    type: Literal["faithfulness"] = "faithfulness"
    content: FaithfulnessContent


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


class ChatStreamEvents(BaseModel):
    """Documentation only: the events of POST /v1/chat/stream, one per `data:` line, as JSON."""

    token: TokenEvent
    sources: SourcesEvent
    done: DoneEvent
    faithfulness: FaithfulnessEvent
    error: ErrorEvent


class RetrievalComparison(BaseModel):
    baseline: List[Source]
    reranked: List[Source]
    baseline_latency_ms: float
    reranked_latency_ms: float
    reranker: Optional[str] = None
    degraded: List[str] = Field(default_factory=list, description="Why the result is not what the configuration asked for.")
