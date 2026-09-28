import asyncio
import datetime
import logging
from typing import Any, Optional

from src.api.models.query_models import QueryRequest, SourceDocument
from src.config import get_settings
from src.retrieving.retriever import DenseRetriever, OptionalReranker
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.models import GenerationResult
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
        if rewriter:
            if get_settings().enable_query_generalisation:
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

    @staticmethod
    async def log_query(
        metrics_store, tenant_id: str, body: QueryRequest, retrieval_result, result: GenerationResult, latency_ms: float
    ) -> Optional[int]:
        """Persists the query's latency, tokens, serving provider and cost. Never fails the request."""
        if not metrics_store:
            return None
        details = {
            "top_k_requested": body.top_k,
            "faithfulness_reasoning": None,
            "retrieved_context": [
                {"chunk_id": c.chunk_id, "source_url": c.source_url, "similarity_score": c.similarity_score}
                for c in result.context_window.included_chunks
            ],
        }
        try:
            return await asyncio.to_thread(
                metrics_store.log_query,
                tenant_id=tenant_id, query=body.query, latency_ms=latency_ms,
                tokens_used=result.prompt_tokens + result.completion_tokens, faithfulness_score=None, details=details,
                embedding_tokens=retrieval_result.embedding_tokens, embedding_cost_usd=retrieval_result.embedding_cost_usd,
                generation_input_tokens=result.prompt_tokens, generation_output_tokens=result.completion_tokens,
                rerank_tokens=retrieval_result.rerank_tokens, provider=result.provider,
                generation_cost_usd=result.generation_cost_usd,
            )
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return None

    @staticmethod
    def evaluate_faithfulness(
        evaluator: FaithfulnessEvaluator, result: GenerationResult, log_id: Optional[int], metrics_store, pipeline_logger
    ) -> Optional[GenerationResult]:
        """Judges the answer against its context and records the score on the query's log row."""
        try:
            evaluated = evaluator.evaluate(result)
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return None
        if log_id and metrics_store:
            try:
                metrics_store.update_faithfulness(log_id, evaluated.faithfulness_score, evaluated.faithfulness_reasoning)
            except Exception as e:
                logger.error(f"Failed to update faithfulness score: {e}")
        if pipeline_logger:
            pipeline_logger.log_event("faithfulness_complete", query_text=result.query, score=evaluated.faithfulness_score)
        return evaluated
