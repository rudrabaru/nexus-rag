"""Test data builders shared by the unit tests."""
from src.retrieving.models import RetrievedChunk


def retrieved_chunk(chunk_id: str = "a", score: float = 1.0, text: str = None, **fields) -> RetrievedChunk:
    """A retrieved chunk whose document and text follow its id unless a test says otherwise."""
    fields.setdefault("source_document", chunk_id)
    fields.setdefault("metadata", {})
    return RetrievedChunk(
        chunk_id=chunk_id, text=text if text is not None else f"text of {chunk_id}", similarity_score=score, **fields
    )
