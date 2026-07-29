"""
RAG Generator: thin LLM API wrapper.

Supported providers: 'gemini' (GEMINI_API_KEY), 'groq' (GROQ_API_KEY).
All intelligence lives in context_builder.py and prompt_template.py.
On API failure returns a GenerationResult with answer="[Generation failed: ...]".
"""

import logging
import time
from typing import List

from .context_builder import ContextBuilder
from .models import GenerationConfig, GenerationResult
from .prompt_template import build_prompt
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    End-to-end RAG generator.

    Orchestrates: retrieve → build context → construct prompt → call LLM → return result.

    Each step is timed and recorded in the GenerationResult for full observability.
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.context_builder = ContextBuilder(self.config)
        self.llm_client = LLMClient(self.config)

    def generate(
        self,
        query: str,
        top_k: int = 5,
        retrieval_result=None,
        chat_history: List[dict] = None,
        stream: bool = False,
    ):
        """Run the full RAG pipeline for a single query. Returns GenerationResult, or a token generator if stream=True."""
        total_start = time.time()

        if retrieval_result is None:
            raise ValueError("retrieval_result must be provided")
            
        retrieval_latency = retrieval_result.latency_ms

        query_log_str = query[:60] + "..." if len(query) > 60 else query
        logger.info(
            f"Retrieved {len(retrieval_result.chunks)} chunks for query: '{query_log_str}'"
        )
        for i, chunk in enumerate(retrieval_result.chunks):
            logger.debug(
                f"  Chunk [{i+1}] score={chunk.similarity_score:.4f} "
                f"id={chunk.chunk_id}"
            )

        context_start = time.time()
        context_window = self.context_builder.build(retrieval_result.chunks)
        context_latency = (time.time() - context_start) * 1000

        included_count = len(context_window.included_chunks)
        excluded_count = len(context_window.excluded_chunks)
        logger.info(
            f"Context: {included_count} chunks included "
            f"(~{context_window.total_context_tokens} tokens), "
            f"{excluded_count} dropped by budget"
        )

        if included_count == 0 and excluded_count > 0:
            scores = [f"{c.similarity_score:.4f}" for c in context_window.excluded_chunks]
            logger.warning(
                f"EMPTY CONTEXT: All {excluded_count} retrieved chunks were excluded. "
                f"min_similarity_score={self.config.min_similarity_score}. "
                f"Excluded scores: [{', '.join(scores[:20])}]"
            )
            diagnostic = (
                f"Retrieval returned no relevant context for this query "
                f"(all {excluded_count} chunks scored below {self.config.min_similarity_score}). "
                f"This may indicate a document quality, embedding, or retrieval configuration issue."
            )
            
            if stream:
                async def _diagnostic_stream():
                    yield diagnostic
                return _diagnostic_stream()
                
            return GenerationResult(
                query=query,
                answer=diagnostic,
                context_window=context_window,
                retrieval_latency_ms=retrieval_latency,
                context_build_latency_ms=context_latency,
                generation_latency_ms=0,
                total_latency_ms=(time.time() - total_start) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
            )

        prompt = build_prompt(
            query, context_window.context_text, self.config, chat_history=chat_history
        )

        gen_start = time.time()

        if stream:
            return self.llm_client.call_llm_stream(prompt)

        answer, raw_response, prompt_tokens, completion_tokens = (
            self.llm_client.call_llm(prompt)
        )
        gen_latency = (time.time() - gen_start) * 1000

        total_latency = (time.time() - total_start) * 1000

        logger.info(
            f"Generated answer in {gen_latency:.1f}ms. Total: {total_latency:.1f}ms"
        )
        logger.info(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        return GenerationResult(
            query=query,
            answer=answer,
            context_window=context_window,
            prompt_used=prompt,
            raw_llm_response=raw_response,
            retrieval_latency_ms=retrieval_latency,
            context_build_latency_ms=context_latency,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=self.config.model_name,
            provider=self.config.provider,
        )
