"""Helpers shared by the eval_integrity tests."""
from src.evaluation.dataset import EvaluationQuery
from src.retrieving.models import RetrievedChunk


def chunk(chunk_id="a", text="some text", url="https://docs.example/page") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_document="Doc", source_url=url, text=text, similarity_score=0.9, token_count=10, metadata={"source_url": url})


def query(text="q", **extra) -> EvaluationQuery:
    return EvaluationQuery(query=text, acceptable_documents=["page"], **extra)
