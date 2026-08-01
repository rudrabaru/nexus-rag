import logging
import datetime
import asyncio
from typing import Optional, Any

from src.api.models.query_models import QueryRequest, SourceDocument
from src.retrieving.retriever import DenseRetriever, OptionalReranker
from src.generating.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)

class QueryService:


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
        import os
        if rewriter:
            if os.environ.get("ENABLE_QUERY_GENERALISATION", "false").lower() == "true":
                search_query = await asyncio.to_thread(
                    rewriter.generalise, search_query
                )
            if body.history:
                search_query = await asyncio.to_thread(
                    rewriter.rewrite, search_query, body.history
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
