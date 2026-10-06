import hashlib
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from src.db.schema import query_embeddings


def query_key(query: str) -> str:
    """The identity of a query as far as its embedding goes: its text, ignoring case and surrounding space."""
    return hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()


class QueryEmbeddingStore:
    """Embeddings of the questions an experiment asked, so a rerun does not spend the provider's rate limit again."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def get(self, index_id: str, key: str) -> Optional[Tuple[List[float], int]]:
        stmt = select(query_embeddings.c.embedding, query_embeddings.c.tokens).where(
            query_embeddings.c.index_id == index_id, query_embeddings.c.query_hash == key
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        return ([float(x) for x in row.embedding], row.tokens) if row else None

    def put(self, index_id: str, key: str, vector: List[float], tokens: int) -> None:
        stmt = insert(query_embeddings).values(index_id=index_id, query_hash=key, embedding=vector, tokens=tokens)
        with self._engine.begin() as conn:
            conn.execute(stmt.on_conflict_do_nothing(index_elements=["index_id", "query_hash"]))
