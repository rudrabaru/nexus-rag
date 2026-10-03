"""
Evaluation datasets: a JSON list of queries, each naming the documents (and optionally the
headings) that would answer it. By default relevance is judged by document and heading, not by
chunk id, so a dataset survives re-chunking and re-embedding, and a query may have several valid
sources. A query may also carry the exact chunks that answer it (source_chunk_ids), which an
experiment uses only when it asks for chunk-level relevance; those ids die with a re-chunk.
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

SYNTHETIC = "synthetic"


class EvaluationQuery(BaseModel):
    query: str
    acceptable_documents: List[str] = Field(min_length=1)
    acceptable_headings: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)  # ground truth for chunk-level relevance
    reference_answer: str = ""
    expected_topic: str = ""
    difficulty: str = "unspecified"
    category: str = "unspecified"
    origin: str = "manual"  # SYNTHETIC for generated queries: say so wherever their metrics are shown
    lexical_overlap: Optional[float] = None  # share of the query's words found in its source chunk (synthetic only)


class Dataset(BaseModel):
    name: str
    content_hash: str  # sha256 of the file: two experiments ran the same queries iff their hashes match
    queries: List[EvaluationQuery]


def load_dataset(path: str, relevance: str = "document") -> Dataset:
    raw = Path(path).read_bytes()
    queries = [EvaluationQuery(**item) for item in json.loads(raw)]
    problems = integrity_problems(queries, relevance)
    if problems:
        raise ValueError("The dataset is not usable:\n  " + "\n  ".join(problems))
    return Dataset(name=Path(path).name, content_hash=hashlib.sha256(raw).hexdigest(), queries=queries)


def integrity_problems(queries: List[EvaluationQuery], relevance: str = "document") -> List[str]:
    """Problems that make metrics meaningless: no queries, duplicates, empty text, or no chunk ground truth."""
    if not queries:
        return ["it contains no queries"]
    problems, seen = [], set()
    for i, q in enumerate(queries):
        text = q.query.strip()
        if not text:
            problems.append(f"query {i} is empty")
        elif text.lower() in seen:
            problems.append(f"query {i} duplicates an earlier query: {text!r}")
        seen.add(text.lower())
        if relevance == "chunk" and not q.source_chunk_ids:
            problems.append(f"query {i} has no source_chunk_ids, which chunk-level relevance needs: {text!r}")
    return problems
