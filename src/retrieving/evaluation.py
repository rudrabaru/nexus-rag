import json
import logging
import statistics
from typing import List

from .evaluation_models import (
    EvaluationQuery,
    EvaluationResult,
    EvaluationReport,
)
from .evaluation_helpers import evaluate_chunk
from .pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, pipeline: RetrievalPipeline, tenant_id: str = None):
        self.pipeline = pipeline
        self.tenant_id = tenant_id

    async def evaluate(self, queries: List[EvaluationQuery]) -> EvaluationReport:
        """
        Runs queries one at a time inside the caller's event loop. Sequential on purpose: the
        report records per-query latency percentiles, which concurrent queries would distort.
        """
        top_k = self.pipeline.config.top_k
        reranked = self.pipeline.reranker is not None
        results = []
        latencies = []
        reranker_failures = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        rr_sum = 0.0

        for q in queries:
            pre_rank = -1
            pre_exact_rank = -1
            result = await self.pipeline.run(q.query, tenant_id=self.tenant_id, allow_global=True)
            if result.degraded:
                logger.warning(f"Degraded retrieval for {q.query!r}: {result.degraded}")
            pre_order = [
                {"chunk_id": c.chunk_id, "source": c.source_document, "score": c.similarity_score}
                for c in result.candidates[:top_k]
            ]
            for idx, c in enumerate(result.candidates):
                _, _, pre_rank, pre_exact_rank, _ = evaluate_chunk(c, q, idx, pre_rank, pre_exact_rank)
            latencies.append(result.latency_ms)

            chunk_infos = []
            best_match_type = "No Match" if not result.chunks else "Wrong Document"
            rank = -1
            exact_match_rank = -1
            retrieved_docs = []

            for i, c in enumerate(result.chunks):
                chunk_info, doc_url, rank, exact_match_rank, match_type = evaluate_chunk(c, q, i, rank, exact_match_rank)
                retrieved_docs.append(doc_url)
                chunk_infos.append(chunk_info)

            if exact_match_rank != -1:
                best_match_type = "Exact Match"
            elif rank != -1:
                best_match_type = "Partial Match"

            if rank != -1:
                if rank <= 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 5:
                    hits_at_5 += 1
                rr_sum += 1.0 / rank

            if reranked and pre_rank != -1:
                # If it was ranked before reranking, and after reranking it is worse (or unranked -1)
                if rank == -1 or rank > pre_rank:
                    post_order = [
                        {"chunk_id": c.chunk_id, "source": c.source_document, "score": c.similarity_score}
                        for c in result.chunks
                    ]
                    promoted = []
                    limit_idx = (rank - 1) if rank != -1 else len(result.chunks)
                    for c in result.chunks[:limit_idx]:
                        promoted.append({
                            "chunk_id": c.chunk_id,
                            "source": c.source_document,
                            "reranker_score": c.similarity_score,
                            "heading_path": c.metadata.get("heading_path", "")
                        })
                    reranker_failures.append({
                        "query": q.query,
                        "expected_target": q.acceptable_documents,
                        "pre_reranking_rank": pre_rank,
                        "post_reranking_rank": rank,
                        "rank_delta": (rank - pre_rank) if rank != -1 else "Unranked (fell off top_k)",
                        "pre_reranking_candidate_order": pre_order,
                        "post_reranking_candidate_order": post_order,
                        "chunks_promoted_above_expected": promoted,
                        "available_source_heading_metadata": [c.metadata for c in result.chunks[:3]]
                    })

            results.append(
                EvaluationResult(
                    query=q.query,
                    expected_topic=q.expected_topic,
                    difficulty=q.difficulty,
                    category=q.category,
                    retrieved_documents=retrieved_docs,
                    retrieved_chunks=chunk_infos,
                    rank=rank,
                    exact_match_rank=exact_match_rank,
                    hit_at_1=(rank == 1),
                    hit_at_3=(1 <= rank <= 3),
                    hit_at_5=(1 <= rank <= 5),
                    best_match_type=best_match_type,
                    latency_ms=result.latency_ms,
                    embedding_latency_ms=result.embedding_latency_ms,
                    search_latency_ms=result.search_latency_ms,
                    rerank_latency_ms=result.rerank_latency_ms,
                    degraded=result.degraded,
                )
            )

        total = len(queries)
        sorted_l = sorted(latencies) if latencies else [0.0]
        p50 = sorted_l[int(len(sorted_l) * 0.50)] if sorted_l else 0.0
        p95 = sorted_l[int(len(sorted_l) * 0.95)] if sorted_l else 0.0

        return EvaluationReport(
            total_queries=total,
            recall_at_1=hits_at_1 / total if total > 0 else 0,
            recall_at_3=hits_at_3 / total if total > 0 else 0,
            recall_at_5=hits_at_5 / total if total > 0 else 0,
            mrr=rr_sum / total if total > 0 else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            reranker_failures=reranker_failures,
            results=results,
        )

    def load_queries_from_json(self, file_path: str) -> List[EvaluationQuery]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append(
                EvaluationQuery(
                    query=item["query"],
                    expected_topic=item.get("expected_topic", "Unknown"),
                    expected_content_type=item.get("expected_content_type", "concept"),
                    acceptable_documents=item.get("acceptable_documents", []),
                    acceptable_headings=item.get("acceptable_headings", []),
                    difficulty=item.get("difficulty", "medium"),
                    category=item.get("category", "Concept"),
                )
            )
        return queries
