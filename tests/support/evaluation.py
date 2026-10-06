"""Helpers shared by the evaluation tests."""
from src.evaluation.dataset import EvaluationQuery
from src.retrieving.models import RetrievedChunk


def chunk(chunk_id, url="https://docs.python.org/3/whatsnew/3.13.html", section="", path=None, score=1.0):
    metadata = {"source_url": url, "section_title": section}
    return RetrievedChunk(
        chunk_id=chunk_id, source_document="Doc", text="t", similarity_score=score, heading_path=list(path or []), metadata=metadata
    )


def query(documents=("3.13.html",), headings=(), **extra):
    return EvaluationQuery(query="q", acceptable_documents=list(documents), acceptable_headings=list(headings), **extra)


def run(rank, **extra):
    return {"query_index": 0, "rank": rank, "latency_ms": 10.0, "degraded": [], "error": None, **extra}
