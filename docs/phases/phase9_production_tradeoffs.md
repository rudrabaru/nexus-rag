# Phase 9: Production Tradeoffs & Architecture Decisions

This document captures the explicit architectural tradeoffs and design decisions made to ensure the RAG system remains highly performant, resilient, and capable of operating as an efficient in-memory microservice.

## 1. API-First Model Architecture
**Decision:** Rely entirely on external APIs for both dense embedding generation and LLM generation, rather than loading local open-weight models.
**Rationale:** Loading modern LLMs and embedding models locally requires massive amounts of RAM and GPU resources. By delegating this compute to specialized external APIs, the core RAG microservice maintains a near-zero memory footprint for ML processing, allowing it to run on extremely resource-constrained infrastructure.
**Tradeoff:** The system introduces a hard dependency on external network calls. Network partitions or API outages will degrade functionality. To mitigate this, the system implements robust exponential backoff and retry logic, and supports seamless fallback providers.

## 2. One Durable Store: Postgres (Neon) with pgvector
**Decision:** Vectors, the keyword index, documents, jobs, API keys and metrics all live in one managed Postgres database (Neon free tier) with the pgvector extension. This replaces Qdrant Cloud for vectors and a local SQLite file for everything else.
**Rationale:**

- *Durability.* The SQLite file sat on an ephemeral disk and was wiped on every restart, taking jobs, keys and cost history with it. Qdrant's free tier deletes clusters after four weeks idle, which is a data-loss failure mode for a portfolio shown on and off for months. Neon suspends idle compute but keeps the data.
- *Consistency.* With two stores, every write was two writes, and they diverged (2,284 vectors against 1,162 keyword rows on the live deployment). In one database a document, its chunks, vectors and keyword entries are written and deleted in one transaction.
- *Explainability.* Retrieval, drill-downs and cost reports become SQL joins over one schema.
**Alternatives considered:** keeping Qdrant (8x the free storage, but idle deletion); a second managed store for metadata (two stores again).
**Tradeoff:** Neon's free tier holds 0.5 GB, about 100k+ chunks at `halfvec(1024)` after index overhead. The compute scales to zero after 5 minutes idle, so the first request after a pause pays a cold start of up to a few seconds (connections are pre-pinged and recycled). A dedicated vector database becomes worth reconsidering at high concurrency (Tier 3 in the redesign plan).

## 3. Keyword Search Inside the Same Rows
**Decision:** Keyword search uses Postgres full-text search over a generated `tsvector` column of the `chunks` table (GIN index) rather than a separate search engine or index.
**Rationale:** A generated column is recomputed by the database on every write, so the keyword index can never lag or diverge from the chunk text. It needs no rebuild step, no second write path and no extra service.
**Tradeoff:** `ts_rank_cd` is not BM25. Hybrid search consumes only rank order through RRF, which limits the impact, but sparse-only rankings can differ from a BM25 engine's. The text-search configuration is `'english'`, so non-English documents are stemmed with English rules; a per-document configuration would fix that. Real BM25 inside Postgres (Neon's `lakebase_bm25`) is a candidate once its free-tier availability is confirmed.

## 3a. Two Database Drivers Over One Schema
**Decision:** One schema definition (`src/registry/schema.py`) serves two engines: asyncpg for query-time search, which runs on the request event loop, and psycopg for registry, job, key and metric writes, which already run in worker threads.
**Rationale:** Search is the latency-critical path and must never block the event loop. The registry is called from roughly 25 synchronous call sites, most of which run in threads. Rewriting them all to async now would collide with the upcoming job-queue work (item 7b), which moves ingestion out of the API process entirely.
**Tradeoff:** Two drivers and two connection pools. Registry calls made directly from async code must be wrapped in `asyncio.to_thread`; the remaining exceptions are progress updates inside the in-process ingestion task, which item 7b removes by moving ingestion to worker processes.

## 3b. Explicit Schema Migrations
**Decision:** Schema changes are Alembic migrations applied as a release step (`alembic upgrade head`). The API never migrates on boot; it checks at startup that the database is at the code's revision and refuses to start otherwise.
**Rationale:** The SQLite schema was created with `CREATE TABLE IF NOT EXISTS` and patched with `ALTER TABLE` inside `try/except`, so nothing recorded which shape a database had. Auto-migrating on boot races when several instances start at once. A failed startup check produces a clear message instead of a failure at the first query that touches a missing column.
**Tradeoff:** Deploying a schema change is two steps (migrate, then roll out). CI runs `alembic check` against real Postgres, so `schema.py` and the migrations cannot drift apart unnoticed.

## 4. Ingestion Concurrency (superseded — see item 13)
**Decision (original):** Enforce strict, semaphore-based concurrency limits on document ingestion, inside the API process.
**Rationale (original):** Ingestion involves memory-intensive tasks like extracting text from massive PDFs and rendering heavy web pages. Allowing unbounded concurrent ingestions would quickly lead to out-of-memory crashes on small servers.
**Superseded by item 13.** Ingestion no longer runs in the API process at all, so an in-process semaphore cannot bound it. The equivalent limit is now the worker's own `Worker(concurrency=N)` (`WORKER_CONCURRENCY`, default 2), which bounds how many jobs one worker process runs at once; running more workers adds capacity without touching the API. The "asynchronous tracking identifier, poll for status" UX this section originally described is unchanged — only how the work behind it is scheduled changed.

## 5. Serverless Web Reading API vs. Local Headless Browser
**Decision:** Utilize an external serverless web reading API for web document crawling and conversion to clean markdown, rather than running a local headless browser or basic HTML scrapers.
**Rationale:** Modern web pages rely heavily on JavaScript (Single Page Applications) and complex DOM structures. Simple HTTP requests and HTML scrapers miss dynamically loaded content and generate noisy boilerplate (navbars, ads, footers). Running a local headless browser requires Chromium dependencies and consumes massive amounts of RAM per tab, which would immediately trigger out-of-memory crashes on resource-constrained environments like free tier hosting (where RAM limits are strictly enforced). Delegating DOM rendering, JS execution, and markdown cleaning to an external reading service maintains a near-zero local memory footprint.
**Tradeoff:** Introduces an external network dependency and rate-limiting constraints from a third-party service. While the reading engine handles anti-bot protection and JavaScript execution effectively, network latency or external service degradation can impact web ingestion latency. To mitigate this, the ingestion pipeline implements exponential backoff retries and structured error guards.

## 6. Rate Limiting Strategy
**Decision:** Apply application-layer rate limiting keyed by the authenticated tenant, falling back to the client IP for anonymous callers. Forwarded client-IP headers (`X-Forwarded-For`, `X-Real-IP`) are honoured only when `TRUST_PROXIES=true`.
**Rationale:** A per-IP limit is spoofable when the API trusts forwarded headers from any caller, so authenticated traffic is limited per tenant, which a client cannot forge. Behind a load balancer the direct peer is the balancer itself, so deployments behind a proxy that overwrites `X-Forwarded-For` should set `TRUST_PROXIES=true`; otherwise anonymous callers share one bucket.
**Tradeoff:** The limiter state is in-process, so with several workers or replicas the effective limit multiplies. A shared store is required before scaling out.

## 7. Stateless Hosts, Durable Database; Hashed, Revocable Keys
**Decision:** The API holds no state on local disk. Everything durable is in Postgres, so an ephemeral host (Render, Hugging Face Spaces) can be wiped or replaced without a recovery step. Tenant API keys are random 256-bit tokens (`nx_...`); only their SHA-256 hash is stored, and each key can be revoked individually (`POST /admin/keys/revoke`, by key or by tenant).
**Rationale:** The previous design worked around an ephemeral disk with two mechanisms that the durable store makes unnecessary. It rebuilt the SQLite registry and keyword index from Qdrant payloads at startup, which only ran when the registry was completely empty, so partial divergence was never repaired. It also used stateless HMAC-signed keys that needed no storage but could not be revoked individually. SHA-256 rather than bcrypt/argon2 is deliberate: slow hashes protect low-entropy passwords, and a 256-bit random token cannot be guessed offline at any hash speed. Keys issued under the old scheme keep working, because the legacy registry already stored `sha256(key)` and the migration imports those hashes; they are revocable from then on.
**Tradeoff:** Every request validates its key, so validation results are cached per process for 60 seconds (a bounded LRU that also caches negative results, so floods of invalid keys cannot become floods of queries). A revocation takes effect immediately in the process that performs it and within 60 seconds elsewhere.

A startup fail-fast guard validates the whole configuration (`src/config.py`) and the schema revision before the server accepts traffic. A failed initialisation makes `/health` return 503, so the platform replaces the instance instead of routing traffic to it.

## 8. Query Generation Concurrency Control
**Decision:** Apply a separate semaphore-based concurrency limit specifically to the query generation pipeline, distinct from the ingestion concurrency limit.
**Rationale:** LLM inference calls during query handling are memory-intensive and incur direct token-based API costs. Allowing unbounded simultaneous queries on a resource-constrained host risks OOM crashes and runaway costs. Excess requests beyond the cap are rejected immediately with **HTTP 503 and `Retry-After`**, not queued. Overload used to be reported as HTTP 200 with the warning in the answer text, which no client or load balancer could tell apart from a real answer.
**Tradeoff:** Under sudden traffic spikes, some requests are explicitly rejected. This is an intentional design decision: a predictable, bounded failure mode is safer and more observable than silent memory exhaustion or runaway billing.

## 9. In-Memory Query Embedding Cache
**Decision:** Cache recently computed query embeddings in a fixed-capacity, MD5-keyed in-memory dictionary.
**Rationale:** Many conversational RAG interactions involve follow-up queries that are semantically similar or even identical to a prior query. Re-embedding the same text via the external Jina API is a wasteful, latency-adding network round-trip. A 500-entry LRU-style in-memory cache eliminates this redundancy for repeated queries at the cost of negligible RAM.
**Tradeoff:** Cache entries do not survive server restarts, and the cache is shared across all tenants (keyed purely on the query string hash). This is acceptable — query text itself is not sensitive, and cache misses simply fall back to a live API call with no correctness impact.

## 10. Admin-Provisioned Workspace Keys
**Decision:** There is no open registration endpoint. Tenant API keys are issued by an administrator through `POST /admin/keys` and revoked through `POST /admin/keys/revoke`, both authenticated with the admin key. The former `POST /register` and `DEMO_MODE` (which mapped every caller to one shared tenant) were removed.
**Rationale:** The earlier model relied on rate limiting as the only abuse control, but `/register` never had a limiter attached, so anyone who found the URL could mint unlimited workspaces. `DEMO_MODE` collapsed tenant isolation entirely. The tenant-isolation mechanism itself is unchanged: every retrieval still filters on `tenant_id` and returns nothing when the tenant is missing.
**Tradeoff:** Onboarding is an administrator action instead of self-service, and the admin key must be handled carefully. This suits a portfolio deployment with a small known set of users. A public product would need real sign-up (OAuth or invite tokens) in front of key issuance.

## 11. Ingest Source Validation
**Decision:** The `url` field of `POST /ingest` accepts only http(s) URLs whose host resolves exclusively to public addresses (`src/ingestion/url_policy.py`).
**Rationale:** The field was passed to the dispatcher unchecked, and the dispatcher treated any string ending in `.pdf`, `.docx`, `.md` or `.txt` as a local file path. Any authenticated tenant could therefore submit a server path and have the file indexed and retrievable. The check is structural (scheme and resolved address); it does not depend on site names.
**Tradeoff:** The check validates the address at request time only. Redirects and DNS rebinding are not covered and are addressed by fetching through a hosted reader service rather than from this process.

## 12. LLM Access: One SDK Call Path, Explicit Retry, Fallback and Timeouts
**Decision:** LiteLLM is used as an SDK (`litellm.completion` / `litellm.acompletion`) for request formatting, response parsing and per-call cost lookup. Retry, fallback and timeouts are one explicit loop in `src/generating/llm_client.py`, not `litellm.Router`.
**Rationale:** When this was built, Router's mid-stream fallback had open bugs (LiteLLM issues #28216 and #40404). Using the plain SDK for both streaming and non-streaming calls keeps one call path. Errors are classified by what a retry can achieve:
- *Transient* (429, 5xx, connection errors, and a 200 with an empty body, which Gemini's thinking mode can produce) are retried with exponential backoff, then fall back.
- *Permanent for this request* (404 dead model, 400, auth errors) are not retried locally, because an identical request cannot succeed; they go straight to the fallback.
- *Timeouts* also go straight to the fallback. Every call is bounded by `request_timeout_seconds` (60 s). LiteLLM's own default is 6,000 s, so without the bound a hung provider held a query slot for up to 100 minutes and the fallback never fired. Observed live on Gemini: a hung call reached the fallback after 218 s before the bound, and after 62 s with it.
- *Streams* fall back only before the first token, because a client that has received part of an answer cannot be handed a different model's answer.
**Tradeoff:** A legitimately slow answer longer than 60 s is cut off and answered by the fallback model instead. Proactive per-deployment rate pacing is not implemented; it is deferred to the configuration-search engine, where call volume requires it.

## 13. Durable Ingestion Queue: Procrastinate, and an API/Worker Split
**Decision:** `POST /ingest` no longer parses anything itself. It validates the request, writes the job (and an uploaded file's bytes) to Postgres, and defers a job onto [Procrastinate](https://procrastinate.readthedocs.io/), a job queue backed by the same database. A separate worker process claims jobs from that queue and does the actual fetching, parsing, chunking and embedding. The API and worker ship as two Docker images (`Dockerfile.api`, `Dockerfile.worker`) from the same source tree.
**Rationale:**
- *What it replaces.* Ingestion previously ran as a FastAPI `BackgroundTasks` callback inside the request-serving process, bounded by an in-process `asyncio.Semaphore`. That queue lived only in memory: a restart lost every queued or in-flight job, and there was no way to run ingestion capacity independently of the web server.
- *Durability without new infrastructure.* Procrastinate's queue is plain Postgres tables, so it needed no new service (unlike Celery or RQ, which need Redis) and inherits the same durability, backups and connection story as everything else in Neon.
- *A real API/worker split, not just a queue.* Splitting into two processes is what makes ingestion capacity scale independently of query-serving capacity, and it is what lets item 8's much heavier worker dependency (Docling) exist without bloating the API image. The API's own dependency list (`requirements-api.txt`) contains no document parser; a container leak check in CI (`docker run ... python -c "import markitdown"`) fails the build if that ever stops being true.
- *Atomicity was the actual bug fix.* Before this, a chunk write, a job-status update and a tenant token-usage update were three separate statements; a crash between them left the job status disagreeing with what was actually stored. `src/jobs/commit.py` makes them one transaction.
- *Locking moved out of application code.* The old code had no protection against two overlapping ingestion runs for the same document. Procrastinate's job-level lock (the document ID) makes the queue itself serialise them, which cannot be forgotten by a code path the way an application-level check could be.
**Recovery of orphaned jobs.** Procrastinate lists jobs whose worker stopped heart-beating but does not requeue them, so a periodic sweep (`src/jobs/recovery.py`) does: requeue while retry budget remains, otherwise fail the job in both the queue and the domain tables. The 30-second threshold is three times the worker's 10-second heartbeat interval, so a heartbeat missed to a Neon compute resume or a long GC pause is not mistaken for a dead worker; the cost is that an orphaned job waits at least that long, plus up to a minute for the next sweep. A worker that was merely paused, not dead, and comes back after its job was requeued is harmless: locks serialise the two runs, chunk writes are idempotent, and a rerun skips any job that already finished.
**Tradeoff:** Uploaded files travel through a Postgres table (`ingest_sources`) rather than a shared filesystem, which is the right call once the API and worker can run on different hosts, but it means Neon's storage now also holds transient upload bytes; a per-tenant cap (`MAX_PENDING_UPLOAD_BYTES`, 200 MB) bounds how much an offline or backlogged worker can accumulate. Two more processes to deploy and keep on the same database revision (both refuse to start on a schema mismatch).
