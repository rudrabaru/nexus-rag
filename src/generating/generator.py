"""
RAG generator: retrieved chunks -> context window -> prompt -> LLM answer.

All intelligence lives in context_builder.py and prompt_template.py; LLM access (retries,
fallback, cost) in llm_client.py. Every step is timed and recorded for observability.
"""
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from .context_builder import ContextBuilder
from .llm_client import LLMCall, LLMClient
from .models import ContextWindow, GenerationConfig, GenerationResult
from .prompt_template import build_prompt

logger = logging.getLogger(__name__)

# Streams whose provider sent no usage block are estimated at ~4 characters per token.
CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class PreparedPrompt:
    query: str
    context_window: ContextWindow
    context_latency_ms: float
    prompt: str = ""
    diagnostic: Optional[str] = None  # set instead of a prompt when no chunk survived context building


class RAGGenerator:
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.context_builder = ContextBuilder(self.config)
        self.llm_client = LLMClient(self.config)

    def prepare(self, query: str, retrieval_result, chat_history: List[dict] = None) -> PreparedPrompt:
        query_log_str = query[:60] + "..." if len(query) > 60 else query
        logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for query: '{query_log_str}'")
        for i, chunk in enumerate(retrieval_result.chunks):
            logger.debug(f"  Chunk [{i+1}] score={chunk.similarity_score:.4f} id={chunk.chunk_id}")

        start = time.time()
        context_window = self.context_builder.build(retrieval_result.chunks)
        prepared = PreparedPrompt(query, context_window, (time.time() - start) * 1000)

        included, excluded = len(context_window.included_chunks), len(context_window.excluded_chunks)
        logger.info(f"Context: {included} chunks included (~{context_window.total_context_tokens} tokens), {excluded} dropped")
        if included == 0 and excluded > 0:
            scores = [f"{c.similarity_score:.4f}" for c in context_window.excluded_chunks]
            logger.warning(
                f"EMPTY CONTEXT: All {excluded} retrieved chunks were excluded. "
                f"min_similarity_score={self.config.min_similarity_score}. Excluded scores: [{', '.join(scores[:20])}]"
            )
            prepared.diagnostic = (
                f"Retrieval returned no relevant context for this query "
                f"(all {excluded} chunks scored below {self.config.min_similarity_score}). "
                f"This may indicate a document quality, embedding, or retrieval configuration issue."
            )
            return prepared

        prepared.prompt = build_prompt(query, context_window.context_text, self.config, chat_history=chat_history)
        return prepared

    def generate(self, query: str, retrieval_result, chat_history: List[dict] = None) -> GenerationResult:
        total_start = time.time()
        prepared = self.prepare(query, retrieval_result, chat_history)
        result = GenerationResult(
            query=query,
            answer=prepared.diagnostic or "",
            context_window=prepared.context_window,
            prompt_used=prepared.prompt,
            retrieval_latency_ms=retrieval_result.latency_ms,
            context_build_latency_ms=prepared.context_latency_ms,
        )
        if prepared.diagnostic:
            result.total_latency_ms = (time.time() - total_start) * 1000
            return result

        gen_start = time.time()
        call = self.llm_client.call_llm(prepared.prompt)
        result.generation_latency_ms = (time.time() - gen_start) * 1000
        result.total_latency_ms = (time.time() - total_start) * 1000
        result.answer = call.text
        result.prompt_tokens, result.completion_tokens = call.prompt_tokens, call.completion_tokens
        result.generation_cost_usd = call.cost_usd
        result.provider, result.model_name = call.provider, call.model

        logger.info(f"Generated answer in {result.generation_latency_ms:.1f}ms. Total: {result.total_latency_ms:.1f}ms")
        logger.info(f"Answer: {call.text[:200]}{'...' if len(call.text) > 200 else ''}")
        return result

    async def stream(self, prepared: PreparedPrompt, call: LLMCall) -> AsyncIterator[str]:
        """Yields the answer as it arrives and fills `call` (owned by the caller) with usage and cost."""
        if prepared.diagnostic:
            call.text = prepared.diagnostic
            yield prepared.diagnostic
            return
        async for piece in self.llm_client.call_llm_stream(prepared.prompt, call):
            yield piece
        if not call.prompt_tokens:
            call.prompt_tokens = len(prepared.prompt) // CHARS_PER_TOKEN_ESTIMATE
        if not call.completion_tokens:
            call.completion_tokens = len(call.text) // CHARS_PER_TOKEN_ESTIMATE
