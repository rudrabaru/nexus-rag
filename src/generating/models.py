"""
Pydantic models for the generation phase.

Separates data shapes from logic to keep all other modules lean and testable.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from src.llm.config import LLMConfig


class GenerationConfig(LLMConfig):
    """
    The generation phase: which model (LLMConfig) plus how much retrieved context it is given.
    Nothing here depends on the corpus.
    """

    max_context_tokens: int = Field(
        5000,
        description=(
            "Maximum tokens of retrieved context in the prompt. With the instructions and chat history "
            "one request is about 5.5K tokens, which fits Groq's free limit of 8K tokens a minute. "
            "Experiment: a search knob (item 13), not tuned on retrieval results."
        ),
    )
    cite_sources: bool = Field(True, description="Instruct the model to cite the [Source: ...] markers of the context")
    min_similarity_score: float = Field(
        0.0,
        description=(
            "Optional minimum similarity score for chunks to be included in context. "
            "Set above 0 only when the score scale is known to be calibrated."
        ),
    )


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
        description="Chunks retrieved but not given to the model: below the score floor, a duplicate of an included chunk, or over the token budget",
    )
    exclusion_reasons: dict = Field(default_factory=dict, description="chunk_id -> why it was excluded: score | duplicate | budget")
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

    # Latency breakdown
    retrieval_latency_ms: float = 0.0
    context_build_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Token accounting
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_cost_usd: float = Field(
        0.0, description="Real per-call cost from litellm.completion_cost(), not an estimate"
    )
    generation_cost_known: bool = Field(True, description="False when litellm has no price for the model: the cost is 0 because unknown, not because free")

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

