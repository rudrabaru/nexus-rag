"""
Chunk search on Postgres, replacing Qdrant and the SQLite FTS5 index.

The dense index (pgvector HNSW) and the sparse index (a generated tsvector with a GIN index)
are columns of the same rows (src/db/schema/), so one write updates both and they
cannot diverge. Writing those rows is src/retrieving/chunk_writes.py; this module is the
search-facing side, used on the request event loop.

Tenant isolation is default-deny, without exception: a missing or wildcard tenant returns
nothing, and no SQL is issued. Evaluations are scoped to one tenant like any other search.

A store is bound to one embedding index (provider:model). Dense and sparse search are both
scoped to it: a query vector is only comparable with vectors of the same model, and keeping
sparse on the same rows means hybrid fusion never mixes two copies of one chunk.
"""
import json
import logging
from typing import List, Optional, Sequence, Tuple

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Text, bindparam, cast, func, literal_column, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from src.db.schema import EMBEDDING_DIMENSION, TEXT_SEARCH_CONFIG, chunks
from src.retrieving.models import RetrievedChunk

logger = logging.getLogger(__name__)

GLOBAL_TENANT_MARKERS = ("ALL", "*")

_TEXT_SEARCH_CONFIG = literal_column(f"'{TEXT_SEARCH_CONFIG}'::regconfig")

_METADATA_COLUMNS = (
    chunks.c.chunk_id,
    chunks.c.tenant_id,
    chunks.c.index_id,
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


def tenant_scope(tenant_id: Optional[str]) -> Optional[str]:
    """The tenant to filter on, or None when the caller must return nothing."""
    if not tenant_id or tenant_id in GLOBAL_TENANT_MARKERS:
        return None
    return tenant_id


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def _to_retrieved_chunk(row, score: float) -> RetrievedChunk:
    metadata = {column.name: row[column.name] for column in _METADATA_COLUMNS}
    # context_builder and src/evaluation/relevance.py parse heading_path as a JSON string, the shape
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
    def __init__(self, sync_engine: Engine, async_engine: AsyncEngine, index_id: str):
        self._sync_engine = sync_engine
        self._async_engine = async_engine
        self.index_id = index_id
        self.distance_metric = "cosine"

    def get_collection_size(self) -> int:
        """Chunks in this store's index, across all tenants."""
        stmt = select(func.count()).select_from(chunks).where(chunks.c.index_id == self.index_id)
        with self._sync_engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    # ── Searches (async, called on the request event loop) ──────────────────

    async def search_dense(
        self, query_embedding: Sequence[float], top_k: int, tenant_id: Optional[str] = None
    ) -> List[RetrievedChunk]:
        tenant = tenant_scope(tenant_id)
        if not tenant or top_k <= 0:
            return []

        query_vector = cast(bindparam("query_vector", vector_literal(query_embedding), type_=Text), HALFVEC(EMBEDDING_DIMENSION))
        distance = chunks.c.embedding.cosine_distance(query_vector)
        stmt = (
            select(*_METADATA_COLUMNS, chunks.c.chunk_text, distance.label("distance"))
            .where(chunks.c.index_id == self.index_id, chunks.c.tenant_id == tenant)
            .order_by(distance, chunks.c.chunk_id)  # chunk_id breaks ties, so a rerun ranks identically
            .limit(top_k)
        )

        async with self._async_engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        # Cosine similarity = 1 - cosine distance. Re-sorted because iterative HNSW scans in
        # relaxed_order may return rows slightly out of distance order.
        results = [_to_retrieved_chunk(row, 1.0 - row["distance"]) for row in rows]
        results.sort(key=lambda c: c.similarity_score, reverse=True)
        return results

    async def search_sparse(
        self, query: str, tenant_id: Optional[str] = None, limit: int = 20
    ) -> Tuple[List[RetrievedChunk], bool]:
        """
        Full-text search. Tries all terms (AND) first for precision, then any term (OR) when
        nothing matches. Returns (chunks, or_fallback_used).
        """
        tenant = tenant_scope(tenant_id)
        if not tenant or limit <= 0 or not query.strip():
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

    async def _run_sparse(self, conn, tsquery, tenant: str, limit: int):
        rank = func.ts_rank_cd(chunks.c.search_vector, tsquery)
        stmt = (
            select(*_METADATA_COLUMNS, chunks.c.chunk_text, rank.label("score"))
            .where(chunks.c.index_id == self.index_id, chunks.c.tenant_id == tenant, chunks.c.search_vector.op("@@")(tsquery))
            .order_by(rank.desc(), chunks.c.chunk_id)  # ts_rank_cd ties constantly: chunk_id makes the order reproducible
            .limit(limit)
        )
        return (await conn.execute(stmt)).mappings().all()
