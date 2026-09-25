"""
Writing chunks to Postgres: row shaping and the upsert statement.

Split out from chunk_store.py (which owns the search-facing ChunkStore class) because this
side runs on a different engine and a different lifecycle: these are plain functions on a
caller-supplied Connection, not methods on ChunkStore, specifically so an ingestion job can
compose a chunk write into the same transaction as its job-status update
(src/jobs/commit.py) — the two can never partially apply.
"""
from typing import Iterable, List

from sqlalchemy import distinct, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from src.embedding.models import EmbeddedChunk
from src.registry.schema import chunks

WRITE_BATCH_SIZE = 100


def _chunk_row(chunk: EmbeddedChunk) -> dict:
    if not chunk.tenant_id or not chunk.doc_id:
        raise ValueError(f"Chunk {chunk.chunk_id} has no tenant_id or doc_id; refusing to store it.")
    return {
        "tenant_id": chunk.tenant_id,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_document": chunk.source_document,
        "source_url": chunk.source_url,
        "title": chunk.title,
        "section_title": chunk.section_title,
        "heading_path": chunk.heading_path or [],
        "content_type": chunk.content_type,
        "contains_code": chunk.contains_code,
        "contains_table": chunk.contains_table,
        "chunk_version": chunk.chunk_version,
        "document_version": chunk.document_version,
        "token_count": chunk.token_count,
        "chunk_text": chunk.chunk_text,
        "embedding": chunk.embedding,
        "embedding_model": chunk.embedding_model,
    }


def chunk_upsert_statement():
    """Insert-or-replace keyed on (tenant_id, chunk_id); generated and defaulted columns are left to Postgres."""
    stmt = insert(chunks)
    updatable = [c.name for c in chunks.columns if c.name not in ("tenant_id", "chunk_id", "search_vector", "created_at")]
    return stmt.on_conflict_do_update(
        index_elements=[chunks.c.tenant_id, chunks.c.chunk_id],
        set_={name: stmt.excluded[name] for name in updatable},
    )


def write_rows(conn: Connection, rows: List[dict]) -> int:
    for start in range(0, len(rows), WRITE_BATCH_SIZE):
        conn.execute(chunk_upsert_statement(), rows[start:start + WRITE_BATCH_SIZE])
    return len(rows)


def write_chunks(conn: Connection, embedded_chunks: Iterable[EmbeddedChunk]) -> int:
    """Upserts chunks with their vectors on the caller's transaction. Idempotent: rewriting replaces rows in place."""
    return write_rows(conn, [_chunk_row(c) for c in embedded_chunks])


def existing_source_urls(conn: Connection, tenant_id: str, doc_id: str) -> set:
    """Source URLs already indexed for a document, so a partially failed sitemap can resume."""
    stmt = select(distinct(chunks.c.source_url)).where(
        chunks.c.tenant_id == tenant_id, chunks.c.doc_id == doc_id, chunks.c.source_url.is_not(None)
    )
    return set(conn.execute(stmt).scalars())
