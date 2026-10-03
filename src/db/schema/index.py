"""
The search index: one row per chunk holding its text, metadata, vector and keyword index together.

- chunks is keyed by (tenant_id, index_id, chunk_id). chunk_id begins with the document's id, so
  two documents never produce the same id (a page ingested alone and again through a sitemap are
  different documents). index_id is the embedding index (provider:model): the same chunk can exist
  once per index, so a corpus can be re-embedded with another model and both compared on the
  same text. Vectors of different models are never compared; every search is scoped to one index.
- chunks.doc_id cascades from documents, so deleting a document removes its chunks, vectors
  and keyword-index entries in one transaction. Nothing can diverge between them.
- search_vector is a generated column: the keyword index cannot fall out of sync with the text.
"""
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    Boolean,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR

from src.db.schema.base import EMBEDDING_DIMENSION, TEXT_SEARCH_CONFIG, Json, metadata, now

embedding_indexes = Table(
    "embedding_indexes",
    metadata,
    Column("index_id", Text, primary_key=True),  # provider:model
    Column("provider", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("dimension", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
)

chunks = Table(
    "chunks",
    metadata,
    Column("tenant_id", Text, nullable=False),
    Column("index_id", Text, ForeignKey("embedding_indexes.index_id"), nullable=False),
    Column("chunk_id", Text, nullable=False),
    Column("doc_id", Text, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True),
    Column("source_document", Text, nullable=False),
    Column("source_url", Text),
    Column("title", Text),
    Column("section_title", Text),
    Column("heading_path", Json, nullable=False, server_default=text("'[]'::jsonb")),
    Column("content_type", Text),
    Column("contains_code", Boolean, nullable=False, server_default="false"),
    Column("contains_table", Boolean, nullable=False, server_default="false"),
    Column("chunk_version", Text),
    Column("document_version", Text),
    Column("token_count", Integer),
    Column("chunk_text", Text, nullable=False),
    Column("embedding", HALFVEC(EMBEDDING_DIMENSION), nullable=False),
    Column("embedding_model", Text, nullable=False),
    Column(
        "search_vector",
        TSVECTOR,
        Computed(f"to_tsvector('{TEXT_SEARCH_CONFIG}', chunk_text)", persisted=True),
    ),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
    PrimaryKeyConstraint("tenant_id", "index_id", "chunk_id"),
    # m=16 / ef_construction=64 are pgvector's defaults, not tuned values. One graph holds every
    # index; searches filter by index_id with iterative scans (src/db/engine.py). With a second
    # large index, a partial HNSW index per index_id would keep graphs model-pure.
    Index(
        "ix_chunks_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "halfvec_cosine_ops"},
    ),
    Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
)
