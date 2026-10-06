from dataclasses import dataclass
from typing import Dict, List

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from src.db.schema import embedding_checkpoints


@dataclass(frozen=True)
class Checkpoint:
    input_hash: str
    embedding: List[float]
    tokens: int


def _as_floats(vector) -> List[float]:
    """The driver returns a list or a pgvector HalfVector, depending on how the type is registered."""
    return [float(x) for x in (vector.to_list() if hasattr(vector, "to_list") else vector)]


def delete_checkpoints(conn: Connection, job_id: str) -> None:
    conn.execute(delete(embedding_checkpoints).where(embedding_checkpoints.c.job_id == job_id))


class CheckpointStore:
    """Vectors already computed for a job, saved as each request group completes (see the schema for why)."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def load(self, job_id: str) -> Dict[str, Checkpoint]:
        stmt = select(
            embedding_checkpoints.c.chunk_id, embedding_checkpoints.c.input_hash,
            embedding_checkpoints.c.embedding, embedding_checkpoints.c.tokens,
        ).where(embedding_checkpoints.c.job_id == job_id)
        with self._engine.connect() as conn:
            return {
                row.chunk_id: Checkpoint(row.input_hash, _as_floats(row.embedding), row.tokens)
                for row in conn.execute(stmt)
            }

    def save(self, job_id: str, rows: List[dict]) -> None:
        """rows: {chunk_id, input_hash, embedding, tokens}. Idempotent."""
        if not rows:
            return
        stmt = insert(embedding_checkpoints)
        stmt = stmt.on_conflict_do_update(
            index_elements=[embedding_checkpoints.c.job_id, embedding_checkpoints.c.chunk_id],
            set_={name: stmt.excluded[name] for name in ("input_hash", "embedding", "tokens")},
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, [{"job_id": job_id, **row} for row in rows])
