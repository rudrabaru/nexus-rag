import json
import logging
import datetime
from typing import Optional, Any
import asyncio
from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.api.auth import get_current_tenant, get_real_ip
from slowapi import Limiter

from src.api.models.query_models import QueryRequest, QueryResponse, SourceDocument
from src.api.dependencies import get_generator, get_retriever, get_reranker, get_evaluator, get_rewriter, get_pipeline_logger
from src.generating.generator import RAGGenerator
from src.retrieving.retriever import DenseRetriever, OptionalReranker
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter
from src.services.query_service import QueryService

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_real_ip)

@router.post("/query", response_model=QueryResponse)
@limiter.limit("15/minute")
async def query_rag(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    generator: RAGGenerator = Depends(get_generator),
    retriever: DenseRetriever = Depends(get_retriever),
    reranker: OptionalReranker = Depends(get_reranker),
    evaluator: FaithfulnessEvaluator = Depends(get_evaluator),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        return QueryResponse(
            answer="Please provide a valid API key (X-API-Key header) to query your private workspace. Generate one at the /register endpoint.",
            sources=[], faithfulness_score=None, faithfulness_reasoning=None, latency_ms=0,
        )

    query_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
    if pipeline_logger:
        pipeline_logger.log_event("query_started", query_text=body.query, tenant_id=tenant_id)

    try:
        registry = getattr(request.app.state, "registry", None)
        if registry:
            doc_count = await asyncio.to_thread(registry.get_doc_count, tenant_id)
            if doc_count == 0:
                return QueryResponse(
                    answer="Your workspace has no documents yet. Please go to the 'Add Source(s)' tab and upload a document or URL first.",
                    sources=[], faithfulness_score=None, faithfulness_reasoning=None, latency_ms=0,
                )

        retrieval_result = await QueryService.run_retrieval(body, retriever, reranker, rewriter, pipeline_logger, tenant_id)

        gen_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
        result = await asyncio.to_thread(
            generator.generate,
            body.query, top_k=body.top_k, retrieval_result=retrieval_result, chat_history=body.history,
        )
        if pipeline_logger:
            pipeline_logger.log_event(
                "generation_complete", query_text=body.query, completion_tokens=result.completion_tokens,
                prompt_tokens=result.prompt_tokens, duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - gen_start) * 1000
            )

        sources = QueryService.construct_sources(result)

        if pipeline_logger:
            pipeline_logger.log_event("query_complete", query_text=body.query, duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - query_start_time) * 1000)

        metrics_store = getattr(request.app.state, "metrics_store", None)
        log_id = None
        if metrics_store:
            details = {
                "top_k_requested": body.top_k, "faithfulness_reasoning": None,
                "retrieved_context": [{"chunk_id": c.chunk_id, "source_url": c.source_url, "similarity_score": c.similarity_score} for c in result.context_window.included_chunks],
            }
            try:
                log_id = metrics_store.log_query(
                    tenant_id=tenant_id, query=body.query, latency_ms=result.total_latency_ms,
                    tokens_used=result.prompt_tokens + result.completion_tokens, faithfulness_score=None,
                    details=details, embedding_tokens=retrieval_result.embedding_tokens,
                    generation_input_tokens=result.prompt_tokens, generation_output_tokens=result.completion_tokens,
                    rerank_tokens=retrieval_result.rerank_tokens, provider=getattr(result, "provider", "gemini")
                )
            except Exception as e:
                logger.error(f"Failed to log query: {e}")

        if body.evaluate_faithfulness:
            background_tasks.add_task(QueryService.run_faithfulness_eval_async, evaluator, result, log_id, metrics_store, pipeline_logger)

        return QueryResponse(
            answer=result.answer, sources=sources, faithfulness_score=None, faithfulness_reasoning=None,
            latency_ms=result.total_latency_ms, latency_breakdown={"retrieval": result.retrieval_latency_ms, "generation": result.generation_latency_ms}
        )
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
@limiter.limit("15/minute")
async def query_rag_stream(
    request: Request,
    body: QueryRequest,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    generator: RAGGenerator = Depends(get_generator),
    retriever: DenseRetriever = Depends(get_retriever),
    reranker: OptionalReranker = Depends(get_reranker),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        return StreamingResponse(iter(["Please provide a valid API key (X-API-Key header) to query your private workspace. Generate one at the /register endpoint."]), media_type="text/event-stream")

    query_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
    if pipeline_logger:
        pipeline_logger.log_event("query_started", query_text=body.query, tenant_id=tenant_id)

    try:
        registry = getattr(request.app.state, "registry", None)
        if registry:
            doc_count = await asyncio.to_thread(registry.get_doc_count, tenant_id)
            if doc_count == 0:
                return StreamingResponse(iter(["Your workspace has no documents yet. Please go to the 'Add Source(s)' tab and upload a document or URL first."]), media_type="text/event-stream")

        retrieval_result = await QueryService.run_retrieval(body, retriever, reranker, rewriter, pipeline_logger, tenant_id)

        gen_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
        async def token_generator():
            async for chunk in generator.generate(
                body.query, top_k=body.top_k, retrieval_result=retrieval_result, chat_history=body.history, stream=True,
            ):
                yield chunk
            if pipeline_logger:
                pipeline_logger.log_event(
                    "generation_complete", query_text=body.query, completion_tokens=0, prompt_tokens=0,
                    duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - gen_start) * 1000
                )
                pipeline_logger.log_event("query_complete", query_text=body.query, duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - query_start_time) * 1000)

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error during query/stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_logs(request: Request, tenant_id: Optional[str] = Depends(get_current_tenant)):
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        return {"queries": [], "summary": {}}
        
    def _fetch_logs():
        with registry._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM observability_logs WHERE tenant_id = ? ORDER BY log_id DESC LIMIT 100", (tenant_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    logs = await asyncio.to_thread(_fetch_logs)
            
    total_queries = len(logs)
    total_cost = sum(log.get("total_cost_usd") or 0.0 for log in logs)
    total_latency = sum(log.get("latency_ms") or 0.0 for log in logs)
    
    summary = {
        "total_queries": total_queries,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query_usd": round(total_cost / total_queries, 6) if total_queries > 0 else 0.0,
        "avg_latency_ms": round(total_latency / total_queries, 2) if total_queries > 0 else 0.0
    }

    return {"queries": logs, "summary": summary}
