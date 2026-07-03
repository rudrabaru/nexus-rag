import json
import logging
import statistics
import re
from typing import List, Dict, Any

from pydantic import BaseModel
from .retriever import DenseRetriever, OptionalReranker

logger = logging.getLogger(__name__)


class EvaluationQuery(BaseModel):
    query: str
    expected_topic: str
    expected_content_type: str = "concept"
    acceptable_documents: List[str]
    acceptable_headings: List[str]
    difficulty: str
    category: str


class RetrievedChunkInfo(BaseModel):
    chunk_id: str
    source_document: str
    similarity_score: float
    text: str
    section_title: str = ""
    heading_path: List[str] = []
    metadata: Dict[str, Any] = {}

    raw_identifier: str = ""
    normalized_identifier: List[str] = []
    matched_target: str = ""
    matching_rule_used: str = ""
    classification_reason: str = ""
    match_type: str = "No Match"


class EvaluationResult(BaseModel):
    query: str
    expected_topic: str
    difficulty: str
    category: str
    retrieved_documents: List[str]
    retrieved_chunks: List[RetrievedChunkInfo]
    rank: int = -1
    exact_match_rank: int = -1
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    best_match_type: str = "No Match"
    latency_ms: float
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0


class EvaluationReport(BaseModel):
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float
    results: List[EvaluationResult]


def normalize_identifier(text: str) -> List[str]:
    # Split by any non-alphanumeric separator
    tokens = re.split(r"[^a-z0-9]+", str(text).lower())
    return [t for t in tokens if t]


def is_sublist(sub: List[str], lst: List[str]) -> bool:
    # Checks if 'sub' is a continuous sublist of 'lst'
    if not sub:
        return True
    if not lst:
        return False
    n, m = len(sub), len(lst)
    for i in range(m - n + 1):
        if lst[i : i + n] == sub:
            return True
    return False


class Evaluator:
    def __init__(self, retriever: DenseRetriever, reranker: OptionalReranker = None):
        self.retriever = retriever
        self.reranker = reranker

    def evaluate(
        self, queries: List[EvaluationQuery], top_k: int = 5
    ) -> EvaluationReport:
        results = []
        latencies = []

        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        rr_sum = 0.0

        for q in queries:
            if self.reranker:
                dense_result = self.retriever.retrieve(q.query, top_k=top_k * 4)
                result = self.reranker.rerank(q.query, dense_result.chunks, top_k=top_k)
                result.embedding_latency_ms = dense_result.embedding_latency_ms
                result.search_latency_ms = dense_result.search_latency_ms
                result.latency_ms += dense_result.latency_ms
            else:
                result = self.retriever.retrieve(q.query, top_k=top_k)
            latencies.append(result.latency_ms)

            chunk_infos = []
            best_match_type = "No Match" if not result.chunks else "Wrong Document"
            rank = -1
            exact_match_rank = -1

            retrieved_docs = []

            for i, c in enumerate(result.chunks):
                doc_url = c.metadata.get("source_url", c.source_document)
                retrieved_docs.append(doc_url)

                # Parse heading path
                raw_path = c.metadata.get("heading_path", "")
                heading_path = []
                if raw_path:
                    try:
                        parsed = json.loads(raw_path)
                        heading_path = (
                            parsed if isinstance(parsed, list) else [str(parsed)]
                        )
                    except (json.JSONDecodeError, TypeError):
                        # Legacy fallback: if stored as " > " joined string
                        heading_path = [
                            s.strip() for s in str(raw_path).split(" > ") if s.strip()
                        ]
                heading_path_str = (
                    " > ".join(heading_path)
                    if isinstance(heading_path, list)
                    else str(heading_path)
                )
                section_title = c.metadata.get("section_title", "")

                raw_identifier = doc_url
                normalized_identifier = normalize_identifier(raw_identifier)

                doc_match = False
                matched_target = ""
                matching_rule_used = ""
                classification_reason = (
                    "No acceptable targets found in retrieved identifiers"
                )

                # 1. Check exact match
                if q.acceptable_documents:
                    for acc in q.acceptable_documents:
                        if (
                            acc in raw_identifier
                        ):  # raw substring match (which works if formatting is perfect)
                            doc_match = True
                            matched_target = acc
                            matching_rule_used = "Exact Substring Match"
                            classification_reason = f"Raw identifier contains exact acceptable target: '{acc}'"
                            break

                    # 2. Check normalized match
                    if not doc_match:
                        for acc in q.acceptable_documents:
                            norm_acc = normalize_identifier(acc)
                            if is_sublist(norm_acc, normalized_identifier):
                                doc_match = True
                                matched_target = acc
                                matching_rule_used = "Normalized Token Sublist Match"
                                classification_reason = f"Normalized identifier contains normalized target tokens: {norm_acc}"
                                break

                # Check heading match
                heading_match = False
                if q.acceptable_headings:
                    for acc_head in q.acceptable_headings:
                        norm_acc_head = normalize_identifier(acc_head)
                        norm_sec_title = normalize_identifier(section_title)
                        norm_head_path = normalize_identifier(heading_path_str)

                        if is_sublist(norm_acc_head, norm_sec_title) or is_sublist(
                            norm_acc_head, norm_head_path
                        ):
                            heading_match = True
                            break

                match_type = "No Match" if not doc_match else "Partial Match"
                if doc_match:
                    if q.acceptable_headings and heading_match:
                        match_type = "Exact Match"
                        classification_reason += " AND Heading matched exactly."
                    elif not q.acceptable_headings:
                        match_type = "Exact Match"
                        classification_reason += " (No heading constraints)."
                    else:
                        match_type = "Partial Match"
                        classification_reason += " BUT Heading constraints failed."

                chunk_infos.append(
                    RetrievedChunkInfo(
                        chunk_id=c.chunk_id,
                        source_document=c.source_document,
                        similarity_score=c.similarity_score,
                        text=c.text,
                        section_title=section_title,
                        heading_path=(
                            heading_path if isinstance(heading_path, list) else []
                        ),
                        metadata=c.metadata,
                        raw_identifier=raw_identifier,
                        normalized_identifier=normalized_identifier,
                        matched_target=matched_target,
                        matching_rule_used=matching_rule_used,
                        classification_reason=classification_reason,
                        match_type=match_type,
                    )
                )

                # Track ranks
                if match_type in ["Exact Match", "Partial Match"] and rank == -1:
                    rank = i + 1
                if match_type == "Exact Match" and exact_match_rank == -1:
                    exact_match_rank = i + 1

            if exact_match_rank != -1:
                best_match_type = "Exact Match"
            elif rank != -1:
                best_match_type = "Partial Match"

            # Score based on document match (rank)
            if rank != -1:
                if rank <= 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 5:
                    hits_at_5 += 1
                rr_sum += 1.0 / rank

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
                    embedding_latency_ms=getattr(result, "embedding_latency_ms", 0.0),
                    search_latency_ms=getattr(result, "search_latency_ms", 0.0),
                    rerank_latency_ms=getattr(result, "rerank_latency_ms", 0.0),
                )
            )

        total = len(queries)
        return EvaluationReport(
            total_queries=total,
            recall_at_1=hits_at_1 / total if total > 0 else 0,
            recall_at_3=hits_at_3 / total if total > 0 else 0,
            recall_at_5=hits_at_5 / total if total > 0 else 0,
            mrr=rr_sum / total if total > 0 else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
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
