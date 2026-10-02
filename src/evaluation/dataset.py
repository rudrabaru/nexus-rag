"""
Evaluation datasets: a JSON list of queries, each naming the documents (and optionally the
headings) that would answer it. Relevance is judged by document and heading, not by chunk id,
so a dataset survives re-chunking and re-embedding, and a query may have several valid sources.
"""
import hashlib
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    query: str
    acceptable_documents: List[str] = Field(min_length=1)
    acceptable_headings: List[str] = Field(default_factory=list)
    expected_topic: str = ""
    difficulty: str = "unspecified"
    category: str = "unspecified"


class Dataset(BaseModel):
    name: str
    content_hash: str  # sha256 of the file: two experiments ran the same queries iff their hashes match
    queries: List[EvaluationQuery]


def load_dataset(path: str) -> Dataset:
    raw = Path(path).read_bytes()
    queries = [EvaluationQuery(**item) for item in json.loads(raw)]
    problems = integrity_problems(queries)
    if problems:
        raise ValueError("The dataset is not usable:\n  " + "\n  ".join(problems))
    return Dataset(name=Path(path).name, content_hash=hashlib.sha256(raw).hexdigest(), queries=queries)


def integrity_problems(queries: List[EvaluationQuery]) -> List[str]:
    """Problems that make metrics meaningless: no queries, duplicates, or empty query text."""
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
    return problems
