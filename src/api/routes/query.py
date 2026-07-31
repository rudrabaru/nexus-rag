import logging
import datetime
import json
from typing import Optional, Any
import asyncio
from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.api.auth import get_current_tenant, get_real_ip
from slowapi import Limiter

from src.api.models.query_models import QueryRequest, QueryResponse
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
            def run_eval_bg(res, lid, m_store, p_logger):
                res = evaluator.evaluate(res)
                if lid and m_store:
                    try:
                        m_store.update_faithfulness(lid, res.faithfulness_score, res.faithfulness_reasoning)
                    except Exception as e:
                        logger.error(f"Failed to update faithfulness score: {e}")
                if p_logger:
                    p_logger.log_event("faithfulness_complete", query_text=res.query, score=res.faithfulness_score)

            background_tasks.add_task(run_eval_bg, result, log_id, metrics_store, pipeline_logger)

        return QueryResponse(
            answer=result.answer, sources=sources, 
            faithfulness_score=None, 
            faithfulness_reasoning=None,
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
    evaluator: FaithfulnessEvaluator = Depends(get_evaluator),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        error_msg = "Please provide a valid API key (X-API-Key header) to query your private workspace. Generate one at the /register endpoint."
        payload = json.dumps({'type': 'token', 'content': error_msg})
        return StreamingResponse(iter([f"data: {payload}\n\n"]), media_type="text/event-stream")

    query_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
    if pipeline_logger:
        pipeline_logger.log_event("query_started", query_text=body.query, tenant_id=tenant_id)

    try:
        query_semaphore = getattr(request.app.state, "query_semaphore", None)
        semaphore_acquired = False
        if query_semaphore:
            if query_semaphore._value <= 0:
                error_msg = "⚠️ **High Traffic Alert:** Our servers are currently at maximum capacity. Please try your query again in a few moments."
                payload = json.dumps({'type': 'token', 'content': error_msg})
                
                async def error_stream():
                    yield f"data: {payload}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
                return StreamingResponse(error_stream(), media_type="text/event-stream")
                
            await query_semaphore.acquire()
            semaphore_acquired = True
            
        try:
            registry = getattr(request.app.state, "registry", None)
            if registry:
                doc_count = await asyncio.to_thread(registry.get_doc_count, tenant_id)
                if doc_count == 0:
                    error_msg = "Your workspace has no documents yet. Please go to the 'Add Source(s)' tab and upload a document or URL first."
                    payload = json.dumps({'type': 'token', 'content': error_msg})
                    return StreamingResponse(iter([f"data: {payload}\n\n"]), media_type="text/event-stream")

                retrieval_result = await QueryService.run_retrieval(body, retriever, reranker, rewriter, pipeline_logger, tenant_id)

                gen_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
                async def token_generator():
                    try:
                        full_answer = ""
                        async for chunk in generator.generate(
                            body.query, top_k=body.top_k, retrieval_result=retrieval_result, chat_history=body.history, stream=True,
                        ):
                            full_answer += chunk
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

                        # Read actual token counts captured during streaming
                        prompt_tokens = getattr(generator.llm_client, "last_prompt_tokens", 0)
                        completion_tokens = getattr(generator.llm_client, "last_completion_tokens", 0)
                        
                        from src.generating.models import GenerationResult
                        context_window = generator.context_builder.build(retrieval_result.chunks)

                        if prompt_tokens == 0:
                            # Fallback heuristic if provider SDK didn't populate usage_metadata on stream chunks
                            prompt_tokens = len(body.query + context_window.context_text) // 4
                        if completion_tokens == 0:
                            completion_tokens = len(full_answer) // 4
                        temp_result = GenerationResult(
                            query=body.query,
                            answer=full_answer,
                            context_window=context_window,
                            retrieval_latency_ms=0,
                            context_build_latency_ms=0,
                            generation_latency_ms=0,
                            total_latency_ms=0,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens
                        )

                        sources = QueryService.construct_sources(temp_result)
                        sources_dict = [s.model_dump() for s in sources]
                        yield f"data: {json.dumps({'type': 'sources', 'content': sources_dict})}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"

                        if pipeline_logger:
                            pipeline_logger.log_event(
                                "generation_complete", query_text=body.query,
                                completion_tokens=completion_tokens, prompt_tokens=prompt_tokens,
                                duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - gen_start) * 1000
                            )
                            pipeline_logger.log_event("query_complete", query_text=body.query, duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - query_start_time) * 1000)

                        metrics_store = getattr(request.app.state, "metrics_store", None)
                        log_id = None
                        if metrics_store:
                            details = {
                                "top_k_requested": body.top_k, "faithfulness_reasoning": None,
                                "retrieved_context": [{"chunk_id": c.chunk_id, "source_url": c.source_url, "similarity_score": c.similarity_score} for c in temp_result.context_window.included_chunks],
                            }
                            try:
                                log_id = metrics_store.log_query(
                                    tenant_id=tenant_id, query=body.query,
                                    latency_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - query_start_time) * 1000,
                                    tokens_used=prompt_tokens + completion_tokens, faithfulness_score=None,
                                    details=details, embedding_tokens=retrieval_result.embedding_tokens,
                                    generation_input_tokens=prompt_tokens, generation_output_tokens=completion_tokens,
                                    rerank_tokens=retrieval_result.rerank_tokens, provider="gemini"
                                )
                            except Exception as e:
                                logger.error(f"Failed to log streaming query: {e}")

                        if body.evaluate_faithfulness:
                            try:
                                eval_res = await asyncio.to_thread(evaluator.evaluate, temp_result)
                                yield f"data: {json.dumps({'type': 'faithfulness', 'content': {'score': eval_res.faithfulness_score, 'reasoning': eval_res.faithfulness_reasoning}})}\n\n"
                                
                                if log_id and metrics_store:
                                    try:
                                        metrics_store.update_faithfulness(log_id, eval_res.faithfulness_score, eval_res.faithfulness_reasoning)
                                    except Exception as e:
                                        logger.error(f"Failed to update faithfulness score: {e}")
                                        
                            except Exception as eval_err:
                                logger.error(f"Async faithfulness evaluation failed: {eval_err}")
                    finally:
                        if query_semaphore:
                            query_semaphore.release()

            semaphore_acquired = False
            return StreamingResponse(token_generator(), media_type="text/event-stream")
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

@router.post("/query/compare")
@limiter.limit("30/minute")
async def compare_retrieval(
    request: Request,
    body: QueryRequest,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    retriever: DenseRetriever = Depends(get_retriever),
    reranker: OptionalReranker = Depends(get_reranker),
    rewriter: Optional[QueryRewriter] = Depends(get_rewriter),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    search_query = body.query
    if rewriter:
        search_query = await asyncio.to_thread(
            rewriter.generalise, search_query
        )
        
    async def get_baseline():
        return await retriever.retrieve(search_query, top_k=body.top_k, tenant_id=tenant_id, pipeline_logger=None)
        
    async def get_reranked():
        dense = await retriever.retrieve(search_query, top_k=body.top_k * 4, tenant_id=tenant_id, pipeline_logger=None)
        if reranker:
            return await reranker.rerank(search_query, dense.chunks, top_k=body.top_k)
        
        # If no reranker is available, just truncate the dense results to top_k to simulate baseline
        dense.chunks = dense.chunks[:body.top_k]
        return dense
        
    baseline_result, reranked_result = await asyncio.gather(get_baseline(), get_reranked())
    
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
            if source_doc and section:
                label = f"{source_doc} > {section}"
            elif source_doc:
                label = source_doc
            else:
                label = section
                
            previews.append({
                "url": chunk.source_url or "",
                "section": label,
                "similarity_score": chunk.similarity_score,
                "chunk_preview": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
            })
        return previews
        
    return {
        "baseline": construct_preview(baseline_result.chunks),
        "reranked": construct_preview(reranked_result.chunks),
        "baseline_latency_ms": baseline_result.latency_ms,
        "reranked_latency_ms": reranked_result.latency_ms + baseline_result.latency_ms
    }
