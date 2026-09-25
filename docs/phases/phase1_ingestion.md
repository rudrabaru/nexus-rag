# Phase 1: Ingestion

## Overview
The ingestion phase serves as the entry point for all raw data entering the Retrieval-Augmented Generation (RAG) system. The core objective is to ingest heterogeneous data formats and normalize them into a uniform, text-based intermediate representation (Markdown) while explicitly preserving structural hierarchies and metadata.

## Core Implementation Logic

### Format Routing
An intelligent routing engine dynamically directs incoming sources to the appropriate processing pipeline:
- **Web URLs**: The router first detects the actual content type. If the URL resolves to a binary document (like a PDF or Word file), it is downloaded and routed to the document extraction engine. Standard web pages are routed to a specialized external serverless web ingestion service.
- **File Uploads**: Files are routed based on their format directly to the document extraction engine.

### Multi-Format Extraction Engine
A unified parsing system converts virtually any document format directly into clean Markdown.

1. **Standard Extraction:** The engine reads text, tables, and lists natively from office documents and PDFs, converting them into structured Markdown.
2. **Multimodal Vision Extraction (Optional):** When requested, a vision-capable AI model intercepts images and scanned pages within the documents. It interprets the visual content and injects text descriptions directly into the Markdown, expanding retrieval capabilities to scanned PDFs and image-heavy presentations.
3. **Structural Post-processing:** The engine automatically recovers structural hierarchy from documents that use typographic conventions instead of native heading styles. It also sanitizes artifacts like tracked changes.

### Web & Sitemap Crawling Engine
For dynamic HTML pages and site hierarchies, the system employs scalable web ingestion adapters:
- **Serverless Web Reading Engine:** Dynamically converts web pages into structured Markdown, stripping unnecessary images and optimizing connection timeouts when visual extraction is disabled to minimize latency.
- **Sitemap Ingestion & Selective Prefix Filtering:** Automatically parses XML sitemaps to discover site pages. To support targeted indexing without full-site deep crawling, the ingestion engine supports URL prefix filtering. When a prefix or subpath is specified, the sitemap parser isolates and ingests only the matching documentation subsections.
- **Bounded Concurrent Crawling:** To maximize ingestion throughput while preventing server overload and API rate-limiting, sitemap URL fetching is executed with bounded asynchronous concurrency.
- **Idempotent Resumption:** Before a sitemap crawl begins, the worker proactively queries Postgres for the source URLs already indexed under the current tenant and document. Previously indexed URLs are skipped entirely, allowing large-scale crawl jobs interrupted by network errors, timeouts, or a worker restart to be safely resumed without re-processing or duplicating content already in the index.

### Durable Job Queue: the API Defers, a Separate Worker Runs
Ingestion runs as two separate processes, each its own container image, connected only through Postgres:
- **The API** validates a request (URL policy, file type, size, tenant quota), registers the job and — for an uploaded file — its raw bytes in the same database transaction, then defers a job onto a Postgres-backed queue ([Procrastinate](https://procrastinate.readthedocs.io/)) and returns immediately. It never parses a document and carries no document-parsing dependency (no MarkItDown, PyMuPDF, Pillow or vision client) — `POST /ingest` returning fast does not depend on how long parsing an 80-page PDF takes.
- **The worker** claims jobs from the queue, fetches or reads the source, parses, chunks and embeds it, then commits the chunks, the job's final status and the tenant's token usage in **one transaction** — a crash between embedding and committing leaves the job "processing" rather than half-written, and re-running is safe because chunk writes are idempotent upserts.

The two processes never share a filesystem, so an uploaded file's bytes travel through Postgres (a small `ingest_sources` table keyed by job ID), not a shared temp directory: the worker reads them back, writes them to its own OS temp directory for the duration of parsing, and the row (and the local file) are gone once the job commits or fails permanently.

**Retry and locking**, both enforced by the queue rather than application code:
- A job that raises an unclassified exception is retried automatically with backoff, up to a fixed budget; a job whose source is fundamentally unprocessable (empty content, oversized) is not retried, since retrying would reproduce the same failure.
- Two jobs for the same document (for example, an accidental double-submit, or a resume retried while the original run is still in flight) share a lock on the document ID, so the queue itself serialises them — the second never starts until the first finishes, instead of both writing that document's chunks at once.

**Recovery from a killed worker.** Procrastinate records which worker holds each running job and that worker's heartbeat, but does not act on a stopped heartbeat. A periodic sweep (once a minute, on any worker) does: an ingestion job whose worker has been silent for 30 seconds goes back on the queue while its retry budget lasts, and is marked failed — in the queue and in the domain tables — once it is spent, so nothing stays "processing" forever. Requeuing is safe because nothing is half-written (the atomic commit above), the uploaded bytes stay in `ingest_sources` until that commit, and chunk writes are idempotent.

A rerun never touches a job that already finished. The failure this prevents: a worker that dies *after* the atomic commit but *before* the queue records success gets requeued, finds its uploaded bytes already deleted, and would otherwise mark a complete document failed.

Measured with a real `docker kill` (SIGKILL) of the worker container at 75% of a 120-chunk ingestion: the document showed 0 chunks and status `pending` (nothing half-written); a fresh worker's sweep requeued the job about 30–40 seconds later; it completed with exactly 120 chunks. Restarting both containers afterwards left the job, its chunks and hybrid retrieval intact with no rebuild step.

### Multi-Tenancy
All documents are tagged at the ingestion layer with:
- **Tenant ID**: Identifies the owning workspace to ensure strict data isolation.
- **Visibility**: Defines the scope of the document (e.g., restricted strictly to the owning tenant).

### Ingestion Observability & Error Propagation
To ensure transparent operations, the ingestion pipeline maintains fine-grained observability over partial failures:
- **Job Status Tracking:** Jobs transition through lifecycle states indicating queuing, active processing, complete success, partial success, or failure.
- **Root Cause Surfacing:** When rate limits, crawling blocks, or embedding timeouts occur on individual pages or batches, the system captures explicit, human-readable error reasons in metadata. These diagnostic messages are propagated directly to the UI, enabling users to inspect exact failure causes even when a job completes with partial success.

### Deduplication
Before starting an extraction job, the system computes a cryptographic hash of the raw file content. If an identical hash already exists in the registry for the same tenant with a completed status, the ingestion is skipped. This prevents redundant reprocessing of identical documents.

## Design Philosophy & Tradeoffs
- **Simplicity vs. Fidelity:** The unified extraction approach favors a fast, uniform conversion layer over highly specialized format parsers. While extremely complex academic layouts might lose some visual context, it significantly reduces pipeline maintenance overhead.
- **Bounded Concurrency vs. Speed:** While unbounded parallel scraping could theoretically process sitemaps faster, enforcing a concurrency limit prevents remote server rate-limiting and ensures stable, predictable memory usage.
- **Crawler Reliability:** JavaScript-heavy sites behind aggressive bot detection may fail to render fully. In these edge cases, explicit error propagation informs the user immediately, allowing manual fallback or selective re-ingestion.
- **Two images instead of one:** Splitting the API and the worker costs an extra process to deploy and a database hand-off for uploaded bytes instead of a shared temp directory. The payoff is that `POST /ingest` cannot be slowed down by parsing, the API's dependency footprint stays small, and ingestion capacity scales independently of query capacity — a large evaluation sweep (item 13) can run more worker replicas without touching the API at all.
- **Application-level locking removed, queue-level locking kept:** Nothing in the ingestion code itself now prevents two jobs for the same document from running at once; that guarantee comes entirely from the job queue's lock, which is simpler to reason about and cannot be bypassed by a code path that forgets to acquire it.
