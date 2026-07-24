import logging
import datetime
import asyncio
from typing import Optional, Any
from fastapi import BackgroundTasks

from src.api.models.query_models import QueryRequest, QueryResponse, SourceDocument
from src.retrieving.retriever import DenseRetriever, OptionalReranker
from src.generating.generator import RAGGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)

class QueryService:
    @staticmethod
    def run_faithfulness_eval_async(evaluator, result, log_id, metrics_store, pipeline_logger):
        try:
            eval_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
            eval_result = evaluator.evaluate(result)
            duration_ms = (datetime.datetime.now(datetime.timezone.utc).timestamp() - eval_start) * 1000
            
            if pipeline_logger:
                pipeline_logger.log_event(
                    "faithfulness_complete",
                    query_text=result.query,
                    score=eval_result.faithfulness_score,
                    duration_ms=duration_ms
                )
                
            if metrics_store and log_id is not None:
                metrics_store.update_faithfulness(
                    log_id, 
                    eval_result.faithfulness_score, 
                    eval_result.faithfulness_reasoning
                )
        except Exception as e:
            logger.error(f"Error in background faithfulness evaluation: {e}")

    @staticmethod
    async def run_retrieval(
        body: QueryRequest,
        retriever: DenseRetriever,
        reranker: OptionalReranker,
        rewriter: Optional[QueryRewriter],
        pipeline_logger: Any,
        tenant_id: str
    ):
        search_query = body.query
        if body.history and rewriter:
            search_query = await asyncio.to_thread(
                rewriter.rewrite, body.query, body.history
            )

        if body.use_reranker:
            ret_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
            dense_result = await retriever.retrieve(
                search_query,
                top_k=body.top_k * 4,
                tenant_id=tenant_id,
                pipeline_logger=pipeline_logger,
            )
            if pipeline_logger:
                pipeline_logger.log_event(
                    "retrieval_complete", 
                    query_text=search_query, 
                    chunk_count=len(dense_result.chunks), 
                    duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - ret_start) * 1000
                )

            rerank_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
            retrieval_result = await reranker.rerank(
                search_query, dense_result.chunks, top_k=body.top_k
            )
            if pipeline_logger:
                pipeline_logger.log_event(
                    "reranking_complete", 
                    query_text=search_query, 
                    chunk_count=len(retrieval_result.chunks), 
                    duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - rerank_start) * 1000
                )
                
            retrieval_result.embedding_latency_ms = dense_result.embedding_latency_ms
            retrieval_result.search_latency_ms = dense_result.search_latency_ms
            retrieval_result.latency_ms += dense_result.latency_ms
        else:
            ret_start = datetime.datetime.now(datetime.timezone.utc).timestamp()
            retrieval_result = await retriever.retrieve(
                search_query,
                top_k=body.top_k,
                tenant_id=tenant_id,
                pipeline_logger=pipeline_logger,
            )
            if pipeline_logger:
                pipeline_logger.log_event(
                    "retrieval_complete", 
                    query_text=search_query, 
                    chunk_count=len(retrieval_result.chunks), 
                    duration_ms=(datetime.datetime.now(datetime.timezone.utc).timestamp() - ret_start) * 1000
                )
        return retrieval_result

    @staticmethod
    def construct_sources(result) -> list[SourceDocument]:
        return [
            SourceDocument(
                url=chunk.source_url or "",
                section=" > ".join(chunk.heading_path) if chunk.heading_path else "",
                similarity_score=chunk.similarity_score,
                chunk_preview=(
                    chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                ),
            )
            for chunk in result.context_window.included_chunks
        ]
