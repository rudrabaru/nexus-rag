from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

MAX_CHAT_TURNS = 20
MAX_TURN_CHARS = 4000  # a long answer pasted back; bounds the prompt a caller can make us pay for


class ChatTurn(BaseModel):
    """One earlier message. Only user and assistant turns exist: a caller cannot inject a "system" turn."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TURN_CHARS)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = False
    evaluate_faithfulness: bool = False
    history: List[ChatTurn] = Field(default_factory=list, max_length=MAX_CHAT_TURNS)

    def history_messages(self) -> List[Dict[str, str]]:
        return [turn.model_dump() for turn in self.history]


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
