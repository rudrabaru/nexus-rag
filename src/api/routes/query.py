import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.api.rate_limit import QUERY_LIMIT, READ_LIMIT, limiter
from src.api.security import require_tenant
from src.api.errors import internal_error
from src.api.dependencies import get_evaluator, get_generator, get_pipeline_logger, get_retrieval, get_rewriter
from src.api.models.query_models import QueryRequest, QueryResponse
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.generator import RAGGenerator
from src.generating.llm_client import LLMCall
from src.generating.models import GenerationResult
from src.generating.query_rewriter import QueryRewriter
from src.retrieving.pipeline import RetrievalResources, build_pipeline
from src.services.query_service import QueryService, chat_config

logger = logging.getLogger(__name__)
router = APIRouter()

EMPTY_WORKSPACE_MESSAGE = (
    "Your workspace has no documents yet. Please go to the 'Add Source(s)' tab and upload a document or URL first."
)
OVERLOAD_RETRY_AFTER_SECONDS = 5


def _reject_when_at_capacity(query_semaphore) -> None:
    """
    Sheds load instead of queueing it. Overload is an HTTP 503 with Retry-After, not an
    HTTP 200 whose answer text says the server is busy, so clients and load balancers can
    tell an overload from an answer.
    """
    if query_semaphore is not None and query_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="Server is at capacity. Retry shortly.",
            headers={"Retry-After": str(OVERLOAD_RETRY_AFTER_SECONDS)},
        )


async def _workspace_is_empty(request: Request, tenant_id: str) -> bool:
    documents = getattr(request.app.state, "documents", None)
    return bool(documents) and await asyncio.to_thread(documents.document_count, tenant_id) == 0


def _sse(event_type: str, content: Any = None) -> str:
    payload = {"type": event_type} if content is None else {"type": event_type, "content": content}
    return f"data: {json.dumps(payload)}\n\n"


def _elapsed_ms(start: float) -> float:
    return (time.time() - start) * 1000


@router.post("/query", response_model=QueryResponse)
@limiter.limit(QUERY_LIMIT)
async def query_rag(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(require_tenant),
    generator: RAGGenerator = Depends(get_generator),
    retrieval: RetrievalResources = Depends(get_retrieval),
    evaluator: FaithfulnessEvaluator = Depends(get_evaluator),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    query_start = time.time()
    if pipeline_logger:
        pipeline_logger.log_event("query_started", query_text=body.query, tenant_id=tenant_id)

    query_semaphore = getattr(request.app.state, "query_semaphore", None)
    _reject_when_at_capacity(query_semaphore)

    try:
        semaphore_acquired = False
        if query_semaphore:
            await query_semaphore.acquire()
            semaphore_acquired = True

        try:
            if await _workspace_is_empty(request, tenant_id):
                return QueryResponse(answer=EMPTY_WORKSPACE_MESSAGE, sources=[], latency_ms=0)

            retrieval_result = await QueryService.run_retrieval(body, retrieval, rewriter, pipeline_logger, tenant_id)

            gen_start = time.time()
            result = await asyncio.to_thread(generator.generate, body.query, retrieval_result, body.history_messages())
            if pipeline_logger:
                pipeline_logger.log_event(
                    "generation_complete", query_text=body.query, completion_tokens=result.completion_tokens,
                    prompt_tokens=result.prompt_tokens, duration_ms=_elapsed_ms(gen_start),
                )
                pipeline_logger.log_event("query_complete", query_text=body.query, duration_ms=_elapsed_ms(query_start))

            log_id = await QueryService.log_query(
                getattr(request.app.state, "query_log", None), tenant_id, body, retrieval_result, result,
                latency_ms=result.total_latency_ms,
            )

            if body.evaluate_faithfulness:
                background_tasks.add_task(
                    QueryService.evaluate_faithfulness, evaluator, result, log_id,
                    getattr(request.app.state, "query_log", None), pipeline_logger,
                )

            return QueryResponse(
                answer=result.answer,
                sources=QueryService.construct_sources(result),
                latency_ms=result.total_latency_ms,
                latency_breakdown={"retrieval": result.retrieval_latency_ms, "generation": result.generation_latency_ms},
            )
        finally:
            if semaphore_acquired and query_semaphore:
                query_semaphore.release()
    except Exception as e:
        raise internal_error("query", e)


@router.post("/query/stream")
@limiter.limit(QUERY_LIMIT)
async def query_rag_stream(
    request: Request,
    body: QueryRequest,
    tenant_id: str = Depends(require_tenant),
    generator: RAGGenerator = Depends(get_generator),
    retrieval: RetrievalResources = Depends(get_retrieval),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    evaluator: FaithfulnessEvaluator = Depends(get_evaluator),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    query_start = time.time()
    if pipeline_logger:
        pipeline_logger.log_event("query_started", query_text=body.query, tenant_id=tenant_id)

    query_semaphore = getattr(request.app.state, "query_semaphore", None)
    _reject_when_at_capacity(query_semaphore)

    try:
        semaphore_acquired = False
        if query_semaphore:
            await query_semaphore.acquire()
            semaphore_acquired = True

        try:
            if await _workspace_is_empty(request, tenant_id):
                return StreamingResponse(iter([_sse("token", EMPTY_WORKSPACE_MESSAGE)]), media_type="text/event-stream")

            retrieval_result = await QueryService.run_retrieval(body, retrieval, rewriter, pipeline_logger, tenant_id)
            prepared = generator.prepare(body.query, retrieval_result, body.history_messages())
            call = LLMCall()  # this request's own usage record; the generator is shared
            gen_start = time.time()
            released = False

            def release_once():
                nonlocal released
                if query_semaphore and not released:
                    released = True
                    query_semaphore.release()

            async def token_generator():
                try:
                    async for piece in generator.stream(prepared, call):
                        yield _sse("token", piece)

                    result = GenerationResult(
                        query=body.query, answer=call.text, context_window=prepared.context_window,
                        prompt_used=prepared.prompt, prompt_tokens=call.prompt_tokens,
                        completion_tokens=call.completion_tokens, generation_cost_usd=call.cost_usd,
                        provider=call.provider, model_name=call.model,
                    )
                    yield _sse("sources", [s.model_dump() for s in QueryService.construct_sources(result)])
                    yield _sse("done")

                    if pipeline_logger:
                        pipeline_logger.log_event(
                            "generation_complete", query_text=body.query, completion_tokens=call.completion_tokens,
                            prompt_tokens=call.prompt_tokens, duration_ms=_elapsed_ms(gen_start),
                        )
                        pipeline_logger.log_event("query_complete", query_text=body.query, duration_ms=_elapsed_ms(query_start))

                    query_log = getattr(request.app.state, "query_log", None)
                    log_id = await QueryService.log_query(
                        query_log, tenant_id, body, retrieval_result, result, latency_ms=_elapsed_ms(query_start),
                    )
                    if body.evaluate_faithfulness:
                        evaluated = await asyncio.to_thread(
                            QueryService.evaluate_faithfulness, evaluator, result, log_id, query_log, pipeline_logger
                        )
                        if evaluated:
                            yield _sse("faithfulness", {
                                "score": evaluated.faithfulness_score, "reasoning": evaluated.faithfulness_reasoning,
                            })
                finally:
                    release_once()

            # Ownership of the semaphore passes to the response. It is released when the
            # generator finishes or closes, or by the response's background task, whichever
            # runs first, so a client that disconnects before streaming starts cannot leak it.
            response = StreamingResponse(
                token_generator(), media_type="text/event-stream", background=BackgroundTask(release_once)
            )
            semaphore_acquired = False
            return response
        finally:
            if semaphore_acquired and query_semaphore:
                query_semaphore.release()

    except Exception as e:
        raise internal_error("query/stream", e)


@router.get("/logs")
@limiter.limit(READ_LIMIT)
async def get_logs(request: Request, tenant_id: str = Depends(require_tenant)):
    query_log = getattr(request.app.state, "query_log", None)
    if not query_log:
        return {"queries": [], "summary": {}}

    logs = await asyncio.to_thread(query_log.recent_queries, tenant_id)
    total_queries = len(logs)
    total_cost = sum(log.get("total_cost_usd") or 0.0 for log in logs)
    total_latency = sum(log.get("latency_ms") or 0.0 for log in logs)
    summary = {
        "total_queries": total_queries,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query_usd": round(total_cost / total_queries, 6) if total_queries > 0 else 0.0,
        "avg_latency_ms": round(total_latency / total_queries, 2) if total_queries > 0 else 0.0,
    }
    return {"queries": logs, "summary": summary}


@router.post("/query/compare")
@limiter.limit(QUERY_LIMIT)
async def compare_retrieval(
    request: Request,
    body: QueryRequest,
    tenant_id: str = Depends(require_tenant),
    retrieval: RetrievalResources = Depends(get_retrieval),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
):
    # One run: "baseline" is the first-stage order of the pool the reranker reordered, so the
    # two columns differ only by the reranking step.
    search_query = await QueryService.search_query(body.model_copy(update={"history": []}), rewriter)
    config = chat_config(body.model_copy(update={"use_reranker": True}))
    reranked_result = await build_pipeline(config, retrieval).run(search_query, tenant_id)
    baseline_chunks = (reranked_result.candidates or reranked_result.chunks)[: body.top_k]
    baseline_latency_ms = reranked_result.latency_ms - reranked_result.rerank_latency_ms

    def construct_preview(chunks):
        previews = []
        for chunk in chunks:
            hpath = chunk.metadata.get("heading_path")
            if isinstance(hpath, str):
                section = hpath
            elif isinstance(hpath, list):
                section = " > ".join([str(h) for h in hpath if h])
            else:
                section = ""
            source_doc = chunk.metadata.get("source_document", "")
            label = f"{source_doc} > {section}" if source_doc and section else source_doc or section
            previews.append({
                "url": chunk.source_url or "",
                "section": label,
                "similarity_score": chunk.similarity_score,
                "chunk_preview": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
            })
        return previews

    return {
        "baseline": construct_preview(baseline_chunks),
        "reranked": construct_preview(reranked_result.chunks),
        "baseline_latency_ms": baseline_latency_ms,
        "reranked_latency_ms": reranked_result.latency_ms,
        "reranker": config.reranker,
        "degraded": reranked_result.degraded,
    }
