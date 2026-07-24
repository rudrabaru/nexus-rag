"""
Pydantic models for the generation phase.

Separates data shapes from logic to keep all other modules lean and testable.
"""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """
    Configuration for the generation phase.

    All parameters are corpus-agnostic and control LLM behavior
    or token budget management only.
    """

    # LLM selection
    provider: str = Field(
        "gemini",
        description="LLM provider: 'gemini' or 'groq'",
    )
    model_name: str = Field(
        "gemini-2.0-flash-lite",
        description="Model identifier passed to the provider API",
    )

    # Token budgeting for context assembly
    max_context_tokens: int = Field(
        5000,
        description=(
            "Maximum tokens to allocate to retrieved context. "
            "Controls how many chunks can be included in the prompt. "
            "Set to 8000 to safely fit within standard 32k-128k context windows while leaving room for generation."
        ),
    )
    max_output_tokens: int = Field(
        4096,
        description="Maximum tokens the LLM may generate in its response",
    )
    temperature: float = Field(
        0.1,
        description=(
            "LLM sampling temperature. Low values (0.0–0.2) reduce hallucinations "
            "and make the model stay closer to retrieved context."
        ),
    )

    # Behavior flags
    cite_sources: bool = Field(
        True,
        description="Instruct the LLM to cite source URLs in its answer",
    )

    fallback_config: Optional[dict] = Field(
        None,
        description="Optional GenerationConfig dictionary to use if primary fails.",
    )

    min_similarity_score: float = Field(
        0.0,
        description=(
            "Optional minimum similarity score for chunks to be included in context. "
            "Set above 0 only when the score scale is known to be calibrated."
        ),
    )

    class Config:
        validate_assignment = True


class ContextChunk(BaseModel):
    """A single retrieved chunk prepared for inclusion in a context window."""

    chunk_id: str
    source_url: str
    heading_path: List[str]
    text: str
    similarity_score: float
    token_estimate: int


class ContextWindow(BaseModel):
    """
    The assembled context passed to the LLM.

    Tracks which chunks were included vs excluded due to token budget limits,
    enabling full observability into what the LLM actually received.
    """

    included_chunks: List[ContextChunk] = Field(default_factory=list)
    excluded_chunks: List[ContextChunk] = Field(
        default_factory=list,
        description="Chunks retrieved but dropped due to token budget",
    )
    total_context_tokens: int = 0
    context_text: str = ""


class GenerationResult(BaseModel):
    """
    Complete output from one RAG generation call.

    Every field is preserved for downstream observability, evaluation,
    and debugging — including the full prompt and raw LLM response.
    """

    query: str
    answer: str

    # Observability fields
    context_window: ContextWindow
    prompt_used: str = Field("", description="Full prompt sent to the LLM")
    raw_llm_response: str = Field("", description="Unmodified LLM response text")

    # Latency breakdown
    retrieval_latency_ms: float = 0.0
    context_build_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Token accounting
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Model used
    model_name: str = ""
    provider: str = ""

    # LLM-as-Judge Evaluation
    faithfulness_score: Optional[float] = Field(
        None,
        description="Score of how well the answer is supported by the context (0.0 to 1.0)",
    )
    faithfulness_reasoning: Optional[str] = Field(
        None, description="LLM judge reasoning for the faithfulness score"
    )
    source_hit: Optional[bool] = Field(
        None,
        description="True if an acceptable document was included in the context window",
    )

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
