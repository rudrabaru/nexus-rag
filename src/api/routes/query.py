import asyncio
import copy
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from starlette.background import BackgroundTask

from src.api.auth import get_current_tenant, get_rate_limit_key
from src.api.dependencies import get_evaluator, get_generator, get_pipeline_logger, get_reranker, get_retriever, get_rewriter
from src.api.models.query_models import QueryRequest, QueryResponse
from src.config import get_settings
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.generator import RAGGenerator
from src.generating.llm_client import LLMCall
from src.generating.models import GenerationResult
from src.generating.query_rewriter import QueryRewriter
from src.retrieving.retriever import HybridRetriever, OptionalReranker
from src.services.query_service import QueryService

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_rate_limit_key)

MISSING_KEY_MESSAGE = (
    "Please provide a valid API key (X-API-Key header) to query your private workspace. "
    "Ask your administrator for a key."
)
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
    registry = getattr(request.app.state, "registry", None)
    return bool(registry) and await asyncio.to_thread(registry.get_doc_count, tenant_id) == 0


def _sse(event_type: str, content: Any = None) -> str:
    payload = {"type": event_type} if content is None else {"type": event_type, "content": content}
    return f"data: {json.dumps(payload)}\n\n"


def _elapsed_ms(start: float) -> float:
    return (time.time() - start) * 1000


@router.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def query_rag(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    generator: RAGGenerator = Depends(get_generator),
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: OptionalReranker = Depends(get_reranker),
    evaluator: FaithfulnessEvaluator = Depends(get_evaluator),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        return QueryResponse(answer=MISSING_KEY_MESSAGE, sources=[], latency_ms=0)

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

            retrieval_result = await QueryService.run_retrieval(body, retriever, reranker, rewriter, pipeline_logger, tenant_id)

            gen_start = time.time()
            result = await asyncio.to_thread(generator.generate, body.query, retrieval_result, body.history)
            if pipeline_logger:
                pipeline_logger.log_event(
                    "generation_complete", query_text=body.query, completion_tokens=result.completion_tokens,
                    prompt_tokens=result.prompt_tokens, duration_ms=_elapsed_ms(gen_start),
                )
                pipeline_logger.log_event("query_complete", query_text=body.query, duration_ms=_elapsed_ms(query_start))

            log_id = await QueryService.log_query(
                getattr(request.app.state, "metrics_store", None), tenant_id, body, retrieval_result, result,
                latency_ms=result.total_latency_ms,
            )

            if body.evaluate_faithfulness:
                background_tasks.add_task(
                    QueryService.evaluate_faithfulness, evaluator, result, log_id,
                    getattr(request.app.state, "metrics_store", None), pipeline_logger,
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
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
@limiter.limit("5/minute")
async def query_rag_stream(
    request: Request,
    body: QueryRequest,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    generator: RAGGenerator = Depends(get_generator),
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: OptionalReranker = Depends(get_reranker),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    evaluator: FaithfulnessEvaluator = Depends(get_evaluator),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        return StreamingResponse(iter([_sse("token", MISSING_KEY_MESSAGE)]), media_type="text/event-stream")

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

            retrieval_result = await QueryService.run_retrieval(body, retriever, reranker, rewriter, pipeline_logger, tenant_id)
            prepared = generator.prepare(body.query, retrieval_result, body.history)
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

                    metrics_store = getattr(request.app.state, "metrics_store", None)
                    log_id = await QueryService.log_query(
                        metrics_store, tenant_id, body, retrieval_result, result, latency_ms=_elapsed_ms(query_start),
                    )
                    if body.evaluate_faithfulness:
                        evaluated = await asyncio.to_thread(
                            QueryService.evaluate_faithfulness, evaluator, result, log_id, metrics_store, pipeline_logger
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
        logger.error(f"Error during query/stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(request: Request, tenant_id: Optional[str] = Depends(get_current_tenant)):
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    metrics_store = getattr(request.app.state, "metrics_store", None)
    if not metrics_store:
        return {"queries": [], "summary": {}}

    logs = await asyncio.to_thread(metrics_store.recent_queries, tenant_id)
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
@limiter.limit("5/minute")
async def compare_retrieval(
    request: Request,
    body: QueryRequest,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: OptionalReranker = Depends(get_reranker),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
):
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    search_query = body.query
    if rewriter and get_settings().enable_query_generalisation:
        search_query = await asyncio.to_thread(rewriter.generalise, search_query)

    # One retrieval serves both columns, so the query is embedded once.
    candidates = await retriever.retrieve(search_query, top_k=body.top_k * 4, tenant_id=tenant_id)
    baseline_result = copy.copy(candidates)
    baseline_result.chunks = candidates.chunks[:body.top_k]
    baseline_result.top_k = body.top_k

    reranked_result = await reranker.rerank(search_query, candidates.chunks, top_k=body.top_k) if reranker else baseline_result

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
        "baseline": construct_preview(baseline_result.chunks),
        "reranked": construct_preview(reranked_result.chunks),
        "baseline_latency_ms": baseline_result.latency_ms,
        "reranked_latency_ms": reranked_result.latency_ms + baseline_result.latency_ms,
    }
