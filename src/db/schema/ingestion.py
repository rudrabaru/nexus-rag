"""Documents and the work that turns sources into them: jobs, the sources waiting for a worker, and the fetch audit."""
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    PrimaryKeyConstraint,
    Table,
    Text,
    text,
)

from src.db.schema.base import EMBEDDING_DIMENSION, Json, metadata, now

documents = Table(
    "documents",
    metadata,
    Column("doc_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("format", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("content_hash", Text),
    Column("stats", Json, nullable=False, server_default=text("'{}'::jsonb")),
    Column("error", Text),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=now()),
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
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
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
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
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
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=now()),
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
    Column("outcome", Text, nullable=False),  # fetched | sitemap | sitemap_child | robots_blocked | denied | failed | quota_exceeded
    Column("provider", Text),
    Column("detail", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=now()),
    Index(None, "tenant_id", "created_at"),
    Index(None, "created_at"),
)

# Embeddings already paid for by a running ingestion. At the card-free Voyage limit one ingestion
# is hours of paced requests, so a failure late in the run must not repeat the early part. Rows
# are keyed by the chunk and carry a hash of exactly what was embedded and into which index, so a
# changed chunker or provider can never reuse a stale vector. They are deleted in the commit that
# stores the chunks, or with the job.
embedding_checkpoints = Table(
    "embedding_checkpoints",
    metadata,
    Column("job_id", Text, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False),
    Column("chunk_id", Text, nullable=False),
    Column("input_hash", Text, nullable=False),
    Column("embedding", HALFVEC(EMBEDDING_DIMENSION), nullable=False),
    Column("tokens", Integer, nullable=False),
    PrimaryKeyConstraint("job_id", "chunk_id"),
)
