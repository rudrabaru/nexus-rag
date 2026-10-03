# Phase 1: Ingestion

## Overview
The ingestion phase serves as the entry point for all raw data entering the Retrieval-Augmented Generation (RAG) system. The core objective is to ingest heterogeneous data formats and normalize them into a uniform, text-based intermediate representation (Markdown) while explicitly preserving structural hierarchies and metadata.

## Core Implementation Logic

### Two Sources, Two Workers
Every source becomes Markdown before processing, but the two kinds of source take different routes, on different processes:

- **Web pages and sitemaps** go to the **fetch worker** (slim; the API image), which reads them through hosted reader APIs and stores the Markdown in Postgres (`fetched_pages`). It then queues the parse step.
- **Uploaded files** (PDF, DOCX, TXT, MD) go straight to the **parse worker** (heavy; `Dockerfile.worker`), which converts them with Docling.
- The **parse worker** then processes both the same way: clean, chunk, embed, commit. It never contacts a website: its only inputs are rows in Postgres.

### Web Fetching: Reader APIs Only
**Rule: no process we host sends a request to a third-party website.** An earlier version ran a crawler (Crawl4AI) inside a Hugging Face Space, and the account was suspended for automated traffic. A crawler on shared hosting makes the host's network the thing target sites see. Fetching is therefore delegated to hosted reader APIs, which fetch from their own networks under their own terms:

1. **Jina Reader, keyless** (primary). About 20 requests/minute per IP, and keyless use does not draw on Jina's one-time token grant. It reads HTML and PDFs by URL, so a remote PDF is never downloaded by us.
2. **Firecrawl** (fallback, only when `FIRECRAWL_API_KEY` is set). 1,000 free credits a month.

The previous code broke this rule in small ways that are now gone: the dispatcher requested `robots.txt` and `/sitemap.xml` from the target site directly, sent `HEAD` requests to detect content types, and downloaded remote PDFs itself.

**robots.txt is always respected.** Jina is asked to check it (`X-Robots-Txt`) and answers HTTP 409 when a page is disallowed (verified 2026-09-26 against a disallowed URL). A disallowed page is recorded as `robots_blocked` and **never retried with another reader**: that would be routing around the site owner's decision. Firecrawl's scrape documentation does not state its robots.txt behaviour, which is why it is second, not first. Tavily Extract, listed in the redesign plan as a third reader, is not wired in for the same reason: its robots.txt behaviour is unverified.

**A page must be readable.** Fewer than 30 words from a reader means a login wall, a bot block or an empty shell, not a short document, and the next reader is tried.

### Sitemaps
Only an explicit sitemap URL (ending in `.xml`, or containing "sitemap") triggers multi-page ingestion; any other URL is exactly one page. The previous dispatcher expanded any page URL into its whole site through `robots.txt` auto-discovery, which spent fetch quota and site traffic the user never asked for.

The sitemap itself is read through Jina Reader, which renders a sitemap's `<loc>` entries as links (verified 2026-09-26); the page URLs are the links in that output. A sitemap index is followed one level deep (up to 10 child sitemaps). With a Firecrawl key, Firecrawl's `map` endpoint is the fallback. Media and archive links are skipped, a `?filter=/docs/` suffix keeps only matching URLs, and at most **50 pages** are taken per job.

### Fetch Policy and Controls
These controls stay in our code because they decide *what we are willing to fetch*, not what content is kept:

- **https only**, and the host must resolve only to public addresses (the SSRF guard).
- **Domain lists**, as configuration, not code: a denylist (default: major social networks and people-search sites) and an optional allowlist mode for a public demo.
- **Per-domain pacing**: one request per domain every 3 seconds (`FETCH_MIN_INTERVAL_SECONDS`) per fetch worker, even through a reader, since the reader fetches from the site on our behalf. The fetch worker runs one job at a time, and 3 seconds also matches keyless Jina's ~20 requests/minute.
- **Daily quota per tenant**: 200 successfully fetched pages per 24 hours (`FETCH_DAILY_PAGE_QUOTA`). The API rejects a URL with HTTP 429 once it is used up, and the fetch worker stops mid-sitemap when it runs out. This protects the shared free reader allowances.
- **Audit trail**: every attempt (fetched, sitemap, robots_blocked, denied, failed, quota_exceeded) is written to `fetch_log` with tenant, job, URL, reader and reason. It has no foreign keys, so it outlives document deletion: abuse stays attributable, and a takedown is one document delete.
- **Only authenticated tenants** can submit URLs (there is no open registration).

The policy is checked twice: by the API when the URL is submitted, and again by the fetch worker before each page, because DNS can change in between.

### Parsing Uploads: Docling
Uploaded PDFs and DOCX files are parsed by [Docling](https://docling.org/): layout analysis, reading order, table structure, and OCR of scanned pages (RapidOCR, ONNX, on CPU). TXT and MD files are read as-is. This replaced MarkItDown, a PyMuPDF text fallback, and page-by-page Gemini vision OCR with a 2-second sleep, which spent the scarcest free resource (LLM requests per day) on a job a local layout model does better.

**Why, with evidence.** The chunking audit (Phase 3) found 97.6% of PDF chunks had no heading path: the old extraction lost document structure. A spike on real documents (2026-09-26, CPU only, 2 threads, a fresh process per document):

| Document | Pages | Docling time | Peak memory | Headings | Tables | Note |
|---|---|---|---|---|---|---|
| bitcoin.pdf | 9 | 79 s | 1.7 GB | 15 | 0 | |
| Attention Is All You Need | 15 | 85 s | 2.3 GB | 28 | 4 | |
| VPC Networking (slides) | 30 | 78 s | 2.6 GB | 11 | 1 | |
| Financial statements | 3 | 54 s | 1.7 GB | 6 | 3 | 4x the text PyMuPDF found (5.5k to 22.6k chars) |
| Scanned presentation | 14 | 119 s | 3.4 GB | 34 | 2 | PyMuPDF: 13 chars. Docling OCR: 6,147 chars, zero LLM calls |
| DOCX files (3) | - | ~1 s | 0.4 GB | 0-17 | 0-5 | |

Model loading adds ~16 s per process. Inside the worker image under Docker Desktop (WSL2) the scanned deck took ~14.6 s/page, so 50 pages is ~730 s plus model load, still inside the timeout. Peak memory was **1.7-3.4 GB**, far below the ~12 GB reported in Docling issue #366, so one-at-a-time parsing fits a 16 GB Hugging Face Space.

**How it runs:**
- **A fresh child process per document**, started by spawn. Docling's memory is not reliably released between documents; a child returns it to the OS on exit, can be killed on a timeout, and an out-of-memory kill takes down only the child.
- **One document at a time per worker**, by a lock, so peak memory is one document's whatever `WORKER_CONCURRENCY` is.
- **Caps**: PDFs over **50 pages** (`DOCLING_MAX_PAGES`) skip Docling, and each parse has a **900 s** timeout (`DOCLING_TIMEOUT_SECONDS`). Rationale: the worst measured rate was ~8.5 s/page, so 50 pages is ~425 s here and ~850 s on a CPU twice as slow.
- **A fallback that keeps the content**: a PDF over the page cap, or one Docling fails on (timeout, crash, OOM), is read with PyMuPDF as plain text. Nothing is dropped, but headings are lost, and the job records `parser: pymupdf` so the loss is visible. A DOCX Docling cannot read fails as unprocessable.
- **Export settings**: `&` and `_` are exported literally (Docling escapes them by default, which put "amp" into the keyword index and broke `snake_case` identifiers), image placeholders are omitted, and page headers and footers that the layout model labels as page furniture are excluded: structural evidence, not a keyword rule.
- **Bold-only section titles** in DOCX files (a line that is entirely bold, does not end like a sentence, and is followed by a blank line) are promoted to headings. Many DOCX files style titles that way, and Docling correctly reports them as paragraphs. Measured: 0 to 4 headings on the n8n notes. The rule's punctuation check had a bug (it read the raw line, which always ends in `**`), now fixed.

The optional "extract visuals" mode (LLM image descriptions) was removed with the vision path. Docling can describe pictures with a local vision model; that is not enabled.

### Durable Job Queue: the API Defers, Workers Run
Ingestion runs as three processes, connected only through Postgres:
- **The API** validates a request (fetch policy and daily page quota for a URL; file type, size and pending-upload bytes for a file; the tenant's chunk quota), registers the job and, for an uploaded file, its raw bytes in the same transaction, then defers a job onto a Postgres-backed queue ([Procrastinate](https://procrastinate.readthedocs.io/)) and returns immediately. A URL goes to the `fetch` queue, a file to the `ingest` queue. The API never fetches or parses anything.
- **The fetch worker** (`python -m src.jobs.workers fetch`, the API image) claims `fetch` jobs, stores each page's Markdown in `fetched_pages`, and defers the `ingest` job. It is an async task so the defer runs on the worker's own event loop.
- **The parse worker** (`python -m src.jobs.workers ingest`, `Dockerfile.worker`) claims `ingest` jobs, reads the source rows, parses, chunks and embeds, then commits the chunks, the job's final status, the tenant's token usage and the deletion of the source rows in **one transaction**. A crash before the commit leaves the job "processing" rather than half-written, and re-running is safe because chunk writes are idempotent upserts.

Each worker registers only its own queue's tasks and refuses to start otherwise, so the parse worker (which runs on hosting that must never contact third-party sites) cannot run a fetch task even by misconfiguration.

The processes never share a filesystem, so sources travel through Postgres: an upload's bytes in `ingest_sources`, fetched pages in `fetched_pages`. The parse worker writes an upload to its own temp directory only while parsing, and both kinds of row are deleted with the commit, or when the job finally fails.

**Retry and locking**, both enforced by the queue rather than application code:
- A job that raises an unclassified exception is retried automatically with backoff, up to a fixed budget; a job whose source is fundamentally unprocessable (empty content, oversized) is not retried, since retrying would reproduce the same failure.
- Two jobs for the same document (for example, an accidental double-submit, or a resume retried while the original run is still in flight) share a lock on the document ID, so the queue itself serialises them — the second never starts until the first finishes, instead of both writing that document's chunks at once.

**Recovery from a killed worker.** Procrastinate records which worker holds each running job and that worker's heartbeat, but does not act on a stopped heartbeat. A periodic sweep (once a minute; each worker sweeps its own queue) does: an ingestion job whose worker has been silent for 30 seconds goes back on the queue while its retry budget lasts, and is marked failed — in the queue and in the domain tables — once it is spent, so nothing stays "processing" forever. Requeuing is safe because nothing is half-written (the atomic commit above), the uploaded bytes stay in `ingest_sources` until that commit, and chunk writes are idempotent.

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

### Re-ingesting replaces a document atomically
Submitting a source again does not delete the existing document. The old chunks keep serving queries while the new job waits for a worker, which matters because workers run on demand and may be off for days. When the new run commits, the pages it read have their old chunks replaced in the same transaction that writes the new ones: chunks it did not rewrite are deleted, but only for pages it actually read, so a sitemap page skipped by the daily quota keeps its chunks. A run with failed chunks (`partial_success`) deletes nothing, so it can never leave a document smaller than it was, and a job that fails outright leaves the document searchable with the error recorded as a note. A resume (`resume=true`) only adds.

Chunk ids begin with the document's id, so two documents never share an id: a page ingested alone and again through a sitemap are different documents with different chunks. The API refuses new work for a tenant that already has 10 jobs queued or running, which bounds what one tenant can pile up while no worker is running.

### Deduplication
An upload is hashed by the API before queueing; a URL source is hashed by the parse worker over its fetched Markdown. If an identical hash already exists for the same tenant with a completed status, nothing is chunked or embedded: the job completes pointing at the existing document (its status shows that document's chunk count, and its metadata records `duplicate_of`). For a URL, the placeholder document registered at submission and its fetched pages are deleted in the same transaction, so a duplicate leaves no empty "pending" document behind.

## Design Philosophy & Tradeoffs
- **Fidelity vs. cost:** Docling recovers structure the old text extraction lost, at 30-120 s and up to 3.4 GB per PDF instead of a few seconds. That cost is why it runs only in the parse worker, one document at a time, with a page cap and a text fallback.
- **Reader APIs vs. control:** Delegating fetching gives up control over rendering and timing, and depends on free allowances (keyless Jina ~20 requests/min; Firecrawl 1,000 pages/month). In exchange no hosted process of ours ever touches a target site, which is what keeps the hosting accounts in good standing.
- **Bounded Concurrency vs. Speed:** While unbounded parallel scraping could theoretically process sitemaps faster, enforcing a concurrency limit prevents remote server rate-limiting and ensures stable, predictable memory usage.
- **Crawler Reliability:** JavaScript-heavy sites behind aggressive bot detection may fail to render fully. In these edge cases, per-page outcomes in the job metadata and in `fetch_log` tell the user which pages failed and why.
- **Two images instead of one:** Splitting the API and the worker costs an extra process to deploy and a database hand-off for uploaded bytes instead of a shared temp directory. The payoff is that `POST /v1/documents` cannot be slowed down by parsing, the API's dependency footprint stays small, and ingestion capacity scales independently of query capacity — a large evaluation sweep (item 13) can run more worker replicas without touching the API at all.
- **Application-level locking removed, queue-level locking kept:** Nothing in the ingestion code itself now prevents two jobs for the same document from running at once; that guarantee comes entirely from the job queue's lock, which is simpler to reason about and cannot be bypassed by a code path that forgets to acquire it.
