"""
RAG generator: retrieved chunks -> context window -> prompt -> LLM answer.

All intelligence lives in context_builder.py and prompt_template.py; LLM access (retries,
fallback, cost) in src/llm/client.py. Every step is timed and recorded for observability.
"""
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from src.llm.client import LLMCall, LLMClient
from src.tokens import estimate_tokens

from .context_builder import BELOW_SCORE, DUPLICATE, OVER_BUDGET, ContextBuilder
from .models import ContextWindow, GenerationConfig, GenerationResult
from .prompt_template import build_prompt

logger = logging.getLogger(__name__)


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
        if included == 0:
            prepared.diagnostic = self._empty_context_diagnostic(context_window)
            logger.warning(f"EMPTY CONTEXT: {prepared.diagnostic}")
            return prepared

        prepared.prompt = build_prompt(query, context_window.context_text, self.config, chat_history=chat_history)
        return prepared

    def _empty_context_diagnostic(self, window: ContextWindow) -> str:
        """Why no model call was made: nothing was retrieved, or every chunk was excluded, and for which reason."""
        if not window.excluded_chunks:
            return "Retrieval returned no chunks for this query, so there is nothing to answer from. The index may be empty or the query unrelated to it."
        counts = {reason: list(window.exclusion_reasons.values()).count(reason) for reason in (BELOW_SCORE, DUPLICATE, OVER_BUDGET)}
        parts = []
        if counts[BELOW_SCORE]:
            parts.append(f"{counts[BELOW_SCORE]} scored below the minimum similarity of {self.config.min_similarity_score}")
        if counts[OVER_BUDGET]:
            parts.append(f"{counts[OVER_BUDGET]} did not fit the {self.config.max_context_tokens}-token context budget")
        if counts[DUPLICATE]:
            parts.append(f"{counts[DUPLICATE]} repeated an earlier chunk")
        return (
            f"All {len(window.excluded_chunks)} retrieved chunks were left out of the context: {'; '.join(parts)}. "
            "This points at the retrieval settings, chunk sizes or the context budget rather than at the model."
        )

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
        call = self.llm_client.call_llm(prepared.prompt)  # raises GenerationError once retries and the fallback are spent
        result.generation_latency_ms = (time.time() - gen_start) * 1000
        result.total_latency_ms = (time.time() - total_start) * 1000
        result.answer = call.text
        result.prompt_tokens, result.completion_tokens = call.prompt_tokens, call.completion_tokens
        result.generation_cost_usd, result.generation_cost_known = call.cost_usd, call.cost_known
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
            call.prompt_tokens = estimate_tokens(prepared.prompt)
        if not call.completion_tokens:
            call.completion_tokens = estimate_tokens(call.text)
