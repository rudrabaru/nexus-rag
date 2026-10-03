"""
Is a dataset's chunk-level ground truth still in the index? Chunk ids are derived from a
document's URL and the chunk's position (chunk_id = md5(url)_chunk_NNN), so re-chunking or
re-ingesting changes them, and a query whose source chunk vanished can never be answered
correctly. Checked before a chunk-level experiment runs, never silently ignored.
"""
from typing import Iterable, List

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.db.schema import chunks


def missing_chunk_ids(engine: Engine, tenant_id: str, chunk_ids: Iterable[str]) -> List[str]:
    """The ids that no chunk of this tenant has (the same chunk exists once per embedding index)."""
    wanted = sorted(set(chunk_ids))
    if not wanted:
        return []
    stmt = select(chunks.c.chunk_id).where(chunks.c.tenant_id == tenant_id, chunks.c.chunk_id.in_(wanted)).distinct()
    with engine.connect() as conn:
        present = set(conn.execute(stmt).scalars())
    return [chunk_id for chunk_id in wanted if chunk_id not in present]
