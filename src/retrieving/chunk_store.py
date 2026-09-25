"""
Chunk search on Postgres, replacing Qdrant and the SQLite FTS5 index.

The dense index (pgvector HNSW) and the sparse index (a generated tsvector with a GIN index)
are columns of the same rows (src/registry/schema.py), so one write updates both and they
cannot diverge. Writing those rows is src/retrieving/chunk_writes.py; this module is the
search-facing side, used on the request event loop.

Tenant isolation is default-deny: a missing or wildcard tenant returns nothing, and no SQL is
issued, unless the caller explicitly opts into a global search (offline evaluation only).
"""
import json
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Text, bindparam, cast, func, literal_column, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from src.embedding.models import EmbeddedChunk
from src.registry.schema import EMBEDDING_DIMENSION, TEXT_SEARCH_CONFIG, chunks
from src.retrieving.chunk_writes import _chunk_row, existing_source_urls, write_rows
from src.retrieving.models import RetrievedChunk

logger = logging.getLogger(__name__)

GLOBAL_TENANT_MARKERS = ("ALL", "*")

_TEXT_SEARCH_CONFIG = literal_column(f"'{TEXT_SEARCH_CONFIG}'::regconfig")

_METADATA_COLUMNS = (
    chunks.c.chunk_id,
    chunks.c.tenant_id,
    chunks.c.doc_id,
    chunks.c.source_document,
    chunks.c.source_url,
    chunks.c.title,
    chunks.c.section_title,
    chunks.c.heading_path,
    chunks.c.content_type,
    chunks.c.contains_code,
    chunks.c.contains_table,
    chunks.c.chunk_version,
    chunks.c.document_version,
)


def resolve_tenant_scope(tenant_id: Optional[str], allow_global: bool) -> Tuple[bool, Optional[str]]:
    """Returns (allowed, tenant filter). allowed=False means the caller must return nothing."""
    if not tenant_id or tenant_id in GLOBAL_TENANT_MARKERS:
        return allow_global, None
    return True, tenant_id


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def _to_retrieved_chunk(row, score: float) -> RetrievedChunk:
    metadata = {column.name: row[column.name] for column in _METADATA_COLUMNS}
    # context_builder and evaluation_helpers parse heading_path as a JSON string, the shape
    # the legacy Qdrant payload used. Changing it would silently break heading matching.
    metadata["heading_path"] = json.dumps(row["heading_path"] or [])
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        source_document=row["source_document"],
        source_url=row["source_url"],
        text=row["chunk_text"],
        similarity_score=float(score),
        metadata=metadata,
    )


class ChunkStore:
    def __init__(self, sync_engine: Engine, async_engine: AsyncEngine):
        self._sync_engine = sync_engine
        self._async_engine = async_engine
        self.distance_metric = "cosine"

    def get_collection_size(self) -> int:
        with self._sync_engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(chunks)).scalar_one()

    def load_chunks(self, embedded_chunks: Iterable[EmbeddedChunk]) -> int:
        """
        Convenience wrapper around chunk_writes.write_chunks, opening its own transaction. For
        a write that must commit atomically with other tables (an ingestion job's status),
        call write_chunks(conn, ...) directly on that shared transaction instead
        (src/jobs/commit.py).

        Rows are built (and validated — see chunk_writes._chunk_row) before opening the
        transaction, so invalid input never costs a database round trip.
        """
        rows = [_chunk_row(c) for c in embedded_chunks]
        if not rows:
            return 0
        with self._sync_engine.begin() as conn:
            return write_rows(conn, rows)

    def get_existing_urls(self, tenant_id: str, doc_id: str) -> set:
        """Convenience wrapper around chunk_writes.existing_source_urls, opening its own connection."""
        with self._sync_engine.connect() as conn:
            return existing_source_urls(conn, tenant_id, doc_id)

    # ── Searches (async, called on the request event loop) ──────────────────

    async def search_dense(
        self, query_embedding: Sequence[float], top_k: int, tenant_id: Optional[str] = None, allow_global: bool = False
    ) -> List[RetrievedChunk]:
        allowed, tenant = resolve_tenant_scope(tenant_id, allow_global)
        if not allowed or top_k <= 0:
            return []

        query_vector = cast(bindparam("query_vector", vector_literal(query_embedding), type_=Text), HALFVEC(EMBEDDING_DIMENSION))
        distance = chunks.c.embedding.cosine_distance(query_vector)
        stmt = select(*_METADATA_COLUMNS, chunks.c.chunk_text, distance.label("distance")).order_by(distance).limit(top_k)
        if tenant:
            stmt = stmt.where(chunks.c.tenant_id == tenant)

        async with self._async_engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        # Cosine similarity = 1 - cosine distance. Re-sorted because iterative HNSW scans in
        # relaxed_order may return rows slightly out of distance order.
        results = [_to_retrieved_chunk(row, 1.0 - row["distance"]) for row in rows]
        results.sort(key=lambda c: c.similarity_score, reverse=True)
        return results

    async def search_sparse(
        self, query: str, tenant_id: Optional[str] = None, limit: int = 20, allow_global: bool = False
    ) -> Tuple[List[RetrievedChunk], bool]:
        """
        Full-text search. Tries all terms (AND) first for precision, then any term (OR) when
        nothing matches. Returns (chunks, or_fallback_used).
        """
        allowed, tenant = resolve_tenant_scope(tenant_id, allow_global)
        if not allowed or limit <= 0 or not query.strip():
            return [], False

        # plainto_tsquery parses free text safely (no query-syntax injection). Its AND form
        # is rewritten to OR by swapping the operator in the tsquery's text representation.
        all_terms = func.plainto_tsquery(_TEXT_SEARCH_CONFIG, bindparam("query", query, type_=Text))
        any_term = cast(func.replace(cast(all_terms, Text), " & ", " | "), TSQUERY)

        async with self._async_engine.connect() as conn:
            rows = await self._run_sparse(conn, all_terms, tenant, limit)
            fallback_used = False
            if not rows:
                rows = await self._run_sparse(conn, any_term, tenant, limit)
                fallback_used = True
        return [_to_retrieved_chunk(row, row["score"]) for row in rows], fallback_used

    @staticmethod
    async def _run_sparse(conn, tsquery, tenant: Optional[str], limit: int):
        rank = func.ts_rank_cd(chunks.c.search_vector, tsquery)
        stmt = (
            select(*_METADATA_COLUMNS, chunks.c.chunk_text, rank.label("score"))
            .where(chunks.c.search_vector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(limit)
        )
        if tenant:
            stmt = stmt.where(chunks.c.tenant_id == tenant)
        return (await conn.execute(stmt)).mappings().all()
