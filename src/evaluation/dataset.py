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
from sqlalchemy.engine import Engine

from src.stores.testsets import FROZEN, TestSetStore

SYNTHETIC = "synthetic"
TEST_SET_PREFIX = "testset:"


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


def resolve_dataset(engine: Engine, tenant_id: str, reference: str, relevance: str = "document") -> Dataset:
    """
    The queries an experiment is scored on. `testset:<name>` is a frozen test set of the tenant in
    Postgres (the normal case: reviewed in the database, immutable, identified by its hash);
    anything else is a dataset file, for CI gates and for sharing a set between databases.
    """
    if not reference.startswith(TEST_SET_PREFIX):
        return load_dataset(reference, relevance)
    name = reference[len(TEST_SET_PREFIX):]
    store = TestSetStore(engine)
    head = store.find(tenant_id, name)
    if head is None:
        raise ValueError(f"Workspace {tenant_id!r} has no test set {name!r}.")
    if head["status"] != FROZEN:
        raise ValueError(f"Test set {name!r} is still a draft: review it, then freeze it (python -m src.testsets freeze).")
    queries = [EvaluationQuery(**q) for q in store.accepted_questions(head["test_set_id"])]
    problems = integrity_problems(queries, relevance)
    if problems:
        raise ValueError("The test set is not usable:\n  " + "\n  ".join(problems))
    return Dataset(name=reference, content_hash=head["content_hash"], queries=queries)


def write_dataset(path: str, queries: List[EvaluationQuery]) -> None:
    """A dataset file the engine can load. Refuses a set it would reject."""
    problems = integrity_problems(queries)
    if problems:
        raise ValueError("The dataset is not usable:\n  " + "\n  ".join(problems))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [q.model_dump() for q in queries]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
