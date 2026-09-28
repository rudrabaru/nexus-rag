"""
The Postgres schema. One MetaData object is shared by the sync engine (registry, jobs,
keys, metrics), the async engine (query-time search) and Alembic (migrations).

Design notes:
- chunks is keyed by (tenant_id, index_id, chunk_id). chunk_id is md5(source_url)_chunk_NNN,
  so two tenants ingesting the same URL produce the same chunk_id (the legacy Qdrant store
  keyed points by chunk_id alone, which let the second tenant's upsert overwrite the first's).
  index_id is the embedding index (provider:model): the same chunk can exist once per index,
  so a corpus can be re-embedded with another model and both compared on the same text.
- embedding_indexes has one row per index. Vectors of different models are never compared;
  every search is scoped to one index.
- chunks.doc_id cascades from documents, so deleting a document removes its chunks, vectors
  and sparse index entries in one transaction. Nothing can diverge between them.
- search_vector is a generated column: the sparse index cannot fall out of sync with the text.
- api_keys uses only portable column types and no foreign keys, so the auth store runs the
  same SQL against SQLite in unit tests. A tenant exists when it holds a key; the tenants
  table only carries usage counters and is populated on first use.
"""
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from pgvector.sqlalchemy import HALFVEC

# One width for every index: voyage-4 (output_dimension=1024), bge-m3 and jina-embeddings-v3
# all produce it. A model with another width needs its own column; providers.py refuses
# mismatched vectors instead of letting the insert fail.
EMBEDDING_DIMENSION = 1024

# 'english' stemming matches the porter tokenizer of the SQLite FTS5 index this replaces.
# Known limitation: non-English corpora are tokenized with English rules.
TEXT_SEARCH_CONFIG = "english"

# Python None is stored as SQL NULL, not the JSON value `null`. With the default, a None
# became 'null'::jsonb, and `'null'::jsonb || '{...}'` is array concatenation, so merged job
# metadata turned into [null, {...}] (caught by the integration suite on real Postgres).
Json = JSONB(none_as_null=True)

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def _now():
    return text("now()")


tenants = Table(
    "tenants",
    metadata,
    Column("tenant_id", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("total_embedding_tokens", BigInteger, nullable=False, default=0),
)

api_keys = Table(
    "api_keys",
    metadata,
    Column("key_hash", Text, primary_key=True),  # sha256 hex of the full key
    Column("tenant_id", Text, nullable=False, index=True),
    Column("key_prefix", Text),  # first characters, for identifying a key without storing it
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("revoked_at", DateTime(timezone=True)),
)

documents = Table(
    "documents",
    metadata,
    Column("doc_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("format", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("visibility", Text, nullable=False, server_default="private"),
    Column("content_hash", Text),
    Column("stats", Json, nullable=False, server_default=text("'{}'::jsonb")),
    Column("error", Text),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=_now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=_now()),
    Index(None, "tenant_id", "content_hash"),
    Index(None, "tenant_id", "source"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("job_id", Text, primary_key=True),
    Column("doc_id", Text, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True),
    Column("status", Text, nullable=False),
    Column("progress_pct", Integer, nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=_now()),
    Column("finished_at", DateTime(timezone=True)),
    Column("error", Text),
    Column("metadata", Json),
)

# Uploaded files waiting for the worker. The API and the worker can run on different hosts, so
# an upload is handed over through the database rather than a local temp file. The row is
# deleted in the same transaction that commits the document, or when the job finally fails.
ingest_sources = Table(
    "ingest_sources",
    metadata,
    Column("job_id", Text, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True),
    Column("tenant_id", Text, nullable=False, index=True),
    Column("filename", Text, nullable=False),
    Column("content", LargeBinary, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=_now()),
)

# Pages fetched by the fetch worker, waiting for the parse worker (the URL counterpart of
# ingest_sources). The parse worker never contacts a website: it reads these rows. Deleted
# with the commit that stores the document's chunks, or when the job finally fails.
fetched_pages = Table(
    "fetched_pages",
    metadata,
    Column("job_id", Text, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False),
    Column("url", Text, nullable=False),
    Column("title", Text),
    Column("markdown", Text, nullable=False),
    Column("provider", Text, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=_now()),
    PrimaryKeyConstraint("job_id", "url"),
)

# Audit trail of every URL a tenant asked us to fetch and what happened. No foreign keys, so
# it outlives document deletion: abuse stays attributable, and it backs the daily page quota.
fetch_log = Table(
    "fetch_log",
    metadata,
    Column("log_id", BigInteger, Identity(), primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("job_id", Text),
    Column("url", Text, nullable=False),
    Column("outcome", Text, nullable=False),  # fetched | sitemap | robots_blocked | denied | failed | quota_exceeded
    Column("provider", Text),
    Column("detail", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=_now()),
    Index(None, "tenant_id", "created_at"),
)

embedding_indexes = Table(
    "embedding_indexes",
    metadata,
    Column("index_id", Text, primary_key=True),  # provider:model
    Column("provider", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("dimension", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=_now()),
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
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=_now()),
    PrimaryKeyConstraint("tenant_id", "index_id", "chunk_id"),
    # m=16 / ef_construction=64 are pgvector's defaults, not tuned values. One graph holds every
    # index; searches filter by index_id with iterative scans (src/registry/engine.py). With a
    # second large index, a partial HNSW index per index_id would keep graphs model-pure.
    Index(
        "ix_chunks_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "halfvec_cosine_ops"},
    ),
    Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
)

query_logs = Table(
    "query_logs",
    metadata,
    Column("log_id", BigInteger, Identity(), primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False, server_default=_now()),
    Column("query", Text, nullable=False),
    Column("latency_ms", Float),
    Column("tokens_used", Integer),
    Column("faithfulness_score", Float),
    Column("details", Json),
    Column("provider", Text),
    Column("embedding_tokens", Integer),
    Column("embedding_cost_usd", Float),
    Column("generation_input_tokens", Integer),
    Column("generation_output_tokens", Integer),
    Column("generation_cost_usd", Float),
    Column("rerank_cost_usd", Float),
    Column("total_cost_usd", Float),
    Index(None, "tenant_id", "log_id"),
)

pipeline_events = Table(
    "pipeline_events",
    metadata,
    Column("event_id", BigInteger, Identity(), primary_key=True),
    Column("event", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False, server_default=_now()),
    Column("tenant_id", Text),
    Column("query_id", Text),
    Column("job_id", Text),
    Column("details", Json),
)
