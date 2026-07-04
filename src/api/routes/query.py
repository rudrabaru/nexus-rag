import json
import logging
import datetime
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from langdetect import detect, LangDetectException

from src.api.models import QueryRequest, QueryResponse, SourceDocument
from src.api.startup import app_state
from src.generating.generator import RAGGenerator
from src.retrieving.retriever import DenseRetriever, OptionalReranker
from src.generating.evaluator import FaithfulnessEvaluator

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/query", response_model=QueryResponse)
@limiter.limit("15/minute")
async def query_rag(request: Request, body: QueryRequest):
    if "init_error" in app_state:
        raise HTTPException(status_code=500, detail=app_state["init_error"])
    if "registry" not in app_state:
        raise HTTPException(
            status_code=503, detail="Service initializing, try again in 15s"
        )

    try:
        generator: RAGGenerator = app_state["generator"]
        retriever: DenseRetriever = app_state["retriever"]
        reranker: OptionalReranker = app_state["reranker"]
        evaluator: FaithfulnessEvaluator = app_state["evaluator"]
        rewriter = app_state.get("rewriter")

        try:
            if detect(body.query) != "en":
                raise HTTPException(
                    status_code=400, detail="Only English queries are supported."
                )
        except LangDetectException:
            pass  # Too short to detect, allow it

        search_query = body.query
        if body.history and rewriter:
            search_query = await asyncio.to_thread(
                rewriter.rewrite, body.query, body.history
            )

        if body.use_reranker:
            dense_result = await asyncio.to_thread(
                retriever.retrieve,
                search_query,
                top_k=body.top_k * 4,
                tenant_id=body.tenant_id,
            )
            retrieval_result = await asyncio.to_thread(
                reranker.rerank, search_query, dense_result.chunks, top_k=body.top_k
            )
            retrieval_result.embedding_latency_ms = dense_result.embedding_latency_ms
            retrieval_result.search_latency_ms = dense_result.search_latency_ms
            retrieval_result.latency_ms += dense_result.latency_ms
            result = await asyncio.to_thread(
                generator.generate,
                body.query,
                top_k=body.top_k,
                retrieval_result=retrieval_result,
                chat_history=body.history,
            )
        else:
            retrieval_result = await asyncio.to_thread(
                retriever.retrieve,
                search_query,
                top_k=body.top_k,
                tenant_id=body.tenant_id,
            )
            result = await asyncio.to_thread(
                generator.generate,
                body.query,
                top_k=body.top_k,
                retrieval_result=retrieval_result,
                chat_history=body.history,
            )

        if body.evaluate_faithfulness:
            result = await asyncio.to_thread(evaluator.evaluate, result)

        sources = [
            SourceDocument(
                url=chunk.source_url,
                section=" > ".join(chunk.heading_path) if chunk.heading_path else "",
                similarity_score=chunk.similarity_score,
                chunk_preview=(
                    chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                ),
            )
            for chunk in result.context_window.included_chunks
        ]

        # Unified Trace Approach: Log to observability
        observability_dir = Path("observability")
        observability_dir.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "query": body.query,
            "tenant_id": body.tenant_id,
            "top_k_requested": body.top_k,
            "answer": result.answer,
            "faithfulness_score": result.faithfulness_score,
            "faithfulness_reasoning": result.faithfulness_reasoning,
            "latency_ms": result.total_latency_ms,
            "tokens_used": result.prompt_tokens + result.completion_tokens,
            "retrieved_context": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_url": chunk.source_url,
                    "similarity_score": chunk.similarity_score,
                    "content": chunk.text,
                }
                for chunk in result.context_window.included_chunks
            ],
        }

        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        log_file = observability_dir / f"chat_logs_{today_str}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            faithfulness_score=result.faithfulness_score,
            faithfulness_reasoning=result.faithfulness_reasoning,
            latency_ms=result.total_latency_ms,
        )
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
@limiter.limit("15/minute")
async def query_rag_stream(request: Request, body: QueryRequest):
    if "init_error" in app_state:
        raise HTTPException(status_code=500, detail=app_state["init_error"])
    if "registry" not in app_state:
        raise HTTPException(
            status_code=503, detail="Service initializing, try again in 15s"
        )

    try:
        generator: RAGGenerator = app_state["generator"]
        retriever = app_state["retriever"]
        reranker: OptionalReranker = app_state["reranker"]
        rewriter = app_state.get("rewriter")

        try:
            if detect(body.query) != "en":
                raise HTTPException(
                    status_code=400, detail="Only English queries are supported."
                )
        except LangDetectException:
            pass  # Too short to detect, allow it

        search_query = body.query
        if body.history and rewriter:
            search_query = await asyncio.to_thread(
                rewriter.rewrite, body.query, body.history
            )

        if body.use_reranker:
            dense_result = await asyncio.to_thread(
                retriever.retrieve,
                search_query,
                top_k=body.top_k * 4,
                tenant_id=body.tenant_id,
            )
            retrieval_result = await asyncio.to_thread(
                reranker.rerank, search_query, dense_result.chunks, top_k=body.top_k
            )
        else:
            retrieval_result = await asyncio.to_thread(
                retriever.retrieve,
                search_query,
                top_k=body.top_k,
                tenant_id=body.tenant_id,
            )

        def token_generator():
            for chunk in generator.generate(
                body.query,
                top_k=body.top_k,
                retrieval_result=retrieval_result,
                chat_history=body.history,
                stream=True,
            ):
                yield chunk

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error during query/stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))
