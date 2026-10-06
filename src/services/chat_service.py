"""
Answering a question over a workspace's documents: the one code path behind both the plain and the
streaming chat endpoints.

    prepare(...)  -> the question's search query and retrieved chunks (or "this workspace is empty")
    answer(...)   -> the complete answer
    events(...)   -> the same answer as a stream of typed events

Splitting prepare from the rest lets the API start a response only after retrieval has succeeded, so
a retrieval failure is an HTTP error, not an error halfway through a stream. This module knows
nothing about HTTP or FastAPI.
"""
import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from src.config import Settings, get_settings
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.generator import RAGGenerator
from src.generating.models import GenerationResult
from src.generating.query_rewriter import QueryRewriter
from src.llm.client import LLMCall
from src.llm.errors import GenerationError
from src.retrieving.pipeline import RetrievalResources, build_pipeline
from src.services.chat_config import chat_retrieval_config
from src.services.chat_log import judge_answer, record_query
from src.services.chat_models import (
    EMPTY_WORKSPACE_MESSAGE,
    GENERATION_FAILED_MESSAGE,
    INTERNAL_ERROR_MESSAGE,
    ChatAnswer,
    ChatQuery,
    Comparison,
    Prepared,
    preview,
    sources_of,
)
from src.stores.documents import DocumentStore
from src.stores.query_log import QueryLogStore
from src.stores.workspace import WorkspaceSettingsStore

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return (time.time() - start) * 1000


class ChatService:
    def __init__(
        self,
        retrieval: RetrievalResources,
        generator: RAGGenerator,
        evaluator: FaithfulnessEvaluator,
        rewriter: Optional[QueryRewriter] = None,
        documents: Optional[DocumentStore] = None,
        query_log: Optional[QueryLogStore] = None,
        workspace: Optional[WorkspaceSettingsStore] = None,
        events: Any = None,
        settings: Optional[Settings] = None,
    ):
        self._retrieval, self._generator, self._evaluator, self._rewriter = retrieval, generator, evaluator, rewriter
        self._documents, self._query_log, self._workspace = documents, query_log, workspace
        self._events = events  # the pipeline event log; optional
        self._settings = settings or get_settings()

    # ── Steps ────────────────────────────────────────────────────────────────

    async def prepare(self, tenant_id: str, chat: ChatQuery) -> Prepared:
        self._log("query_started", query_text=chat.query, tenant_id=tenant_id)
        prepared = Prepared(tenant_id, chat, started=time.time())
        if self._documents is not None and await asyncio.to_thread(self._documents.document_count, tenant_id) == 0:
            prepared.empty = True
            return prepared

        workspace = await asyncio.to_thread(self._workspace.get_retrieval, tenant_id) if self._workspace else None
        prepared.config = chat_retrieval_config(self._settings, workspace, chat.top_k, chat.use_reranker)
        query = await self._search_query(chat)
        started = time.time()
        prepared.retrieval = await build_pipeline(prepared.config, self._retrieval).run(
            query, tenant_id, pipeline_logger=self._events
        )
        self._log(
            "retrieval_complete", query_text=query, tenant_id=tenant_id, strategy=prepared.config.strategy,
            reranker=prepared.config.reranker, chunk_count=len(prepared.retrieval.chunks),
            degraded=prepared.retrieval.degraded, duration_ms=_elapsed_ms(started),
        )
        return prepared

    async def answer(self, prepared: Prepared) -> ChatAnswer:
        """The complete answer. Raises GenerationError when the model cannot answer."""
        if prepared.empty:
            return ChatAnswer(answer=EMPTY_WORKSPACE_MESSAGE)
        chat, started = prepared.chat, time.time()
        result = await asyncio.to_thread(
            self._generator.generate, chat.query, prepared.retrieval, list(chat.history)
        )
        self._log("generation_complete", query_text=chat.query, completion_tokens=result.completion_tokens,
                  prompt_tokens=result.prompt_tokens, duration_ms=_elapsed_ms(started))
        self._log("query_complete", query_text=chat.query, duration_ms=_elapsed_ms(prepared.started))
        log_id = await record_query(self._query_log, prepared, result, result.total_latency_ms)
        return ChatAnswer(
            answer=result.answer, sources=sources_of(result), latency_ms=result.total_latency_ms,
            latency_breakdown={"retrieval": result.retrieval_latency_ms, "generation": result.generation_latency_ms},
            result=result, log_id=log_id,
        )

    async def events(self, prepared: Prepared) -> AsyncIterator[Dict[str, Any]]:
        """
        The answer as events: token*, sources, done, then optionally faithfulness. A failure while
        streaming ends the stream with an error event (the HTTP status is already sent by then).
        done means the answer is complete; the faithfulness check can take seconds more.
        """
        if prepared.empty:
            yield {"type": "token", "content": EMPTY_WORKSPACE_MESSAGE}
            yield {"type": "sources", "content": []}
            yield {"type": "done"}
            return

        chat = prepared.chat
        generation = self._generator.prepare(chat.query, prepared.retrieval, list(chat.history))
        call = LLMCall()  # this request's own usage record; the generator is shared
        started = time.time()
        try:
            async for piece in self._generator.stream(generation, call):
                yield {"type": "token", "content": piece}
        except GenerationError:
            logger.exception("Generation failed while streaming")
            yield {"type": "error", "code": "generation_failed", "message": GENERATION_FAILED_MESSAGE}
            return
        except Exception:
            reference = uuid.uuid4().hex[:12]
            logger.exception(f"Streaming failed | reference={reference}")
            yield {"type": "error", "code": "internal_error", "message": f"{INTERNAL_ERROR_MESSAGE} (reference {reference})"}
            return

        result = GenerationResult(
            query=chat.query, answer=call.text, context_window=generation.context_window, prompt_used=generation.prompt,
            prompt_tokens=call.prompt_tokens, completion_tokens=call.completion_tokens,
            generation_cost_usd=call.cost_usd, provider=call.provider, model_name=call.model,
        )
        yield {"type": "sources", "content": [vars(s) for s in sources_of(result)]}
        yield {"type": "done", "latency_ms": _elapsed_ms(prepared.started)}

        self._log("generation_complete", query_text=chat.query, completion_tokens=call.completion_tokens,
                  prompt_tokens=call.prompt_tokens, duration_ms=_elapsed_ms(started))
        self._log("query_complete", query_text=chat.query, duration_ms=_elapsed_ms(prepared.started))
        log_id = await record_query(self._query_log, prepared, result, _elapsed_ms(prepared.started))
        if chat.evaluate_faithfulness:
            judged = await asyncio.to_thread(self.judge, result, log_id)
            if judged:
                yield {"type": "faithfulness", "content": {
                    "score": judged.faithfulness_score, "reasoning": judged.faithfulness_reasoning}}

    async def compare(self, tenant_id: str, chat: ChatQuery) -> "Comparison":
        """
        The retrieval with and without the reranker, from one run: "baseline" is the first-stage
        order of the pool the reranker reordered, so the two differ only by the reranking step.
        """
        workspace = await asyncio.to_thread(self._workspace.get_retrieval, tenant_id) if self._workspace else None
        config = chat_retrieval_config(self._settings, workspace, chat.top_k, use_reranker=True)
        query = await self._search_query(ChatQuery(query=chat.query))
        reranked = await build_pipeline(config, self._retrieval).run(query, tenant_id)
        baseline = (reranked.candidates or reranked.chunks)[: chat.top_k]
        return Comparison(
            baseline=[preview(c) for c in baseline], reranked=[preview(c) for c in reranked.chunks],
            baseline_latency_ms=reranked.latency_ms - reranked.rerank_latency_ms, reranked_latency_ms=reranked.latency_ms,
            reranker=config.reranker, degraded=list(reranked.degraded),
        )

    def judge(self, result: GenerationResult, log_id: Optional[int]) -> Optional[GenerationResult]:
        """Judges the answer against its context and records the score. Never raises."""
        judged = judge_answer(self._evaluator, self._query_log, result, log_id)
        if judged:
            self._log("faithfulness_complete", query_text=result.query, score=judged.faithfulness_score)
        return judged

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _search_query(self, chat: ChatQuery) -> str:
        """The text actually searched: optionally generalised, and made standalone when there is history."""
        query = chat.query
        if self._rewriter and self._settings.enable_query_generalisation:
            query = await asyncio.to_thread(self._rewriter.generalise, query)
        if self._rewriter and chat.history:
            query = await asyncio.to_thread(self._rewriter.rewrite, query, list(chat.history))
        return query

    def _log(self, event: str, **fields) -> None:
        if self._events:
            self._events.log_event(event, **fields)
