# Phase 9: Production Tradeoffs & Architecture Decisions

This document captures the explicit architectural tradeoffs and design decisions made to ensure the RAG system remains highly performant, resilient, and capable of operating as an efficient in-memory microservice.

## 1. API-First Model Architecture
**Decision:** Rely on external APIs for embeddings and LLM generation rather than loading local open-weight models. The one exception is the default reranker: a ~4 MB ONNX cross-encoder (FlashRank) that runs on CPU inside the API, with no torch and no GPU (Phase 5).
**Rationale:** Loading modern LLMs and embedding models locally requires massive amounts of RAM and GPU resources. By delegating this compute to specialized external APIs, the core RAG microservice keeps a small memory footprint and runs on extremely resource-constrained infrastructure. The reranker is small enough to be the exception, and running it locally removes a per-query API cost and a one-time-grant dependency.
**Tradeoff:** The system introduces a hard dependency on external network calls. Network partitions or API outages will degrade functionality. To mitigate this, the system implements robust exponential backoff and retry logic, and supports seamless fallback providers.

## 2. One Durable Store: Postgres (Neon) with pgvector
**Decision:** Vectors, the keyword index, documents, jobs, API keys and metrics all live in one managed Postgres database (Neon free tier) with the pgvector extension. This replaces Qdrant Cloud for vectors and a local SQLite file for everything else.
**Rationale:**

- *Durability.* The SQLite file sat on an ephemeral disk and was wiped on every restart, taking jobs, keys and cost history with it. Qdrant's free tier deletes clusters after four weeks idle, which is a data-loss failure mode for a portfolio shown on and off for months. Neon suspends idle compute but keeps the data.
- *Consistency.* With two stores, every write was two writes, and they diverged (2,284 vectors against 1,162 keyword rows on the live deployment). In one database a document, its chunks, vectors and keyword entries are written and deleted in one transaction.
- *Explainability.* Retrieval, drill-downs and cost reports become SQL joins over one schema.
**Alternatives considered:** keeping Qdrant (8x the free storage, but idle deletion); a second managed store for metadata (two stores again).
**Tradeoff:** Neon's free tier holds 1 GB (confirm the current figure in the Neon console). A `halfvec(1024)` vector is about 2 KB per chunk, before the chunk text and index overhead; the realistic capacity has not been measured on a real corpus. The compute scales to zero after 5 minutes idle, so the first request after a pause pays a cold start of up to a few seconds (connections are pre-pinged and recycled). A dedicated vector database becomes worth reconsidering at high concurrency (Tier 3 in the redesign plan).

## 3. Keyword Search Inside the Same Rows
**Decision:** Keyword search uses Postgres full-text search over a generated `tsvector` column of the `chunks` table (GIN index) rather than a separate search engine or index.
**Rationale:** A generated column is recomputed by the database on every write, so the keyword index can never lag or diverge from the chunk text. It needs no rebuild step, no second write path and no extra service.
**Tradeoff:** `ts_rank_cd` is not BM25. Hybrid search consumes only rank order through RRF, which limits the impact, but sparse-only rankings can differ from a BM25 engine's. The text-search configuration is `'english'`, so non-English documents are stemmed with English rules; a per-document configuration would fix that. Real BM25 inside Postgres (Neon's `lakebase_bm25`) is a candidate once its free-tier availability is confirmed.

## 3a. Two Database Drivers Over One Schema
**Decision:** One schema definition (`src/db/schema/`) serves two engines: asyncpg for query-time search, which runs on the request event loop, and psycopg for registry, job, key and metric writes, which already run in worker threads.
**Rationale:** Search is the latency-critical path and must never block the event loop. The stores (`src/stores/`) are called from synchronous code, mostly in worker threads, and ingestion runs in worker processes, so an async rewrite of them has no measured benefit at this scale.
**Tradeoff:** Two drivers and two connection pools. A store called directly from async code must be wrapped in `asyncio.to_thread`.

## 3b. Explicit Schema Migrations
**Decision:** Schema changes are Alembic migrations applied as a release step (`alembic upgrade head`). The API never migrates on boot; it checks at startup that the database is at the code's revision and refuses to start otherwise.
**Rationale:** A schema created with `CREATE TABLE IF NOT EXISTS` and patched in place records nothing about which shape a database has. Auto-migrating on boot races when several instances start at once. A failed startup check produces a clear message instead of a failure at the first query that touches a missing column.
**Tradeoff:** Deploying a schema change is two steps (migrate, then roll out). The integration suite runs `alembic check` against real Postgres (`tests/integration/test_pg_schema.py`), so the schema definition and the migrations cannot drift apart unnoticed. The startup check requires the exact revision, so during a rolling deploy an instance on the older code refuses to start against a database already migrated; migrations that are safe to run under old code are not yet distinguished.

## 4. Ingestion Concurrency
**Decision:** Bound ingestion by the worker's own concurrency (`WORKER_CONCURRENCY`, default 2, Procrastinate's `Worker(concurrency=N)`), not inside the API.
**Rationale:** Ingestion is memory-intensive (Docling parses one document at a time per worker, Phase 1). It runs only in worker processes, so the limit lives there; more workers add capacity without touching the API. Callers get a job id at once and poll its status.

## 5. Serverless Web Reading API vs. Local Headless Browser
**Decision:** Utilize an external serverless web reading API for web document crawling and conversion to clean markdown, rather than running a local headless browser or basic HTML scrapers.
**Rationale:** Modern web pages rely heavily on JavaScript (Single Page Applications) and complex DOM structures. Simple HTTP requests and HTML scrapers miss dynamically loaded content and generate noisy boilerplate (navbars, ads, footers). Running a local headless browser requires Chromium dependencies and consumes massive amounts of RAM per tab, which would immediately trigger out-of-memory crashes on resource-constrained environments like free tier hosting (where RAM limits are strictly enforced). Delegating DOM rendering, JS execution, and markdown cleaning to an external reading service maintains a near-zero local memory footprint.
**Tradeoff:** Introduces an external network dependency and rate-limiting constraints from a third-party service. While the reading engine handles anti-bot protection and JavaScript execution effectively, network latency or external service degradation can impact web ingestion latency. To mitigate this, the ingestion pipeline implements exponential backoff retries and structured error guards.

## 6. Rate Limiting Strategy
**Decision:** Apply application-layer rate limiting keyed by the authenticated tenant, falling back to the client IP for anonymous callers. `X-Forwarded-For` is believed only for the number of reverse-proxy hops configured in `TRUSTED_PROXY_HOPS` (default 0): the client is the entry that many places from the right, because every entry further left is client-supplied. Repeated rejected credentials from one address are throttled, and request bodies are size-limited before parsing or authentication.
**Rationale:** A per-IP limit is spoofable when the API trusts forwarded headers from any caller, so authenticated traffic is limited per tenant, which a client cannot forge. Behind a load balancer the direct peer is the balancer itself, so a deployment behind one load balancer (Render) sets `TRUSTED_PROXY_HOPS=1`; otherwise anonymous callers share one bucket.
**Tradeoff:** The limiter state is in-process, so with several workers or replicas the effective limit multiplies. A shared store is required before scaling out.

## 7. Stateless Hosts, Durable Database; Hashed, Revocable Keys
**Decision:** The API holds no state on local disk. Everything durable is in Postgres, so an ephemeral host can be wiped or replaced without a recovery step. Tenant API keys are random 256-bit tokens (`nx_...`); only their SHA-256 hash is stored, and each key can be revoked individually (`POST /v1/admin/keys/revoke`, by key or by tenant).
**Rationale:** Keeping all state in the database removes any rebuild-at-startup step that could leave a store half-populated, and stored hashes (rather than stateless signed keys) are what make individual revocation possible. SHA-256 rather than bcrypt/argon2 is deliberate: slow hashes protect low-entropy passwords, and a 256-bit random token cannot be guessed offline at any hash speed.
**Tradeoff:** Every request validates its key, so validation results are cached per process for 60 seconds (a bounded LRU that also caches negative results, so floods of invalid keys cannot become floods of queries). A revocation takes effect immediately in the process that performs it and within 60 seconds elsewhere.

A startup fail-fast guard validates the whole configuration (`src/config.py`) and the schema revision before the server accepts traffic. A failed initialisation makes `/health` return 503, so the platform replaces the instance instead of routing traffic to it.

## 8. Query Generation Concurrency Control
**Decision:** Apply a separate semaphore-based concurrency limit specifically to the query generation pipeline, distinct from the ingestion concurrency limit.
**Rationale:** LLM inference calls during query handling are memory-intensive and incur direct token-based API costs. Allowing unbounded simultaneous queries on a resource-constrained host risks OOM crashes and runaway costs. Excess requests beyond the cap are rejected immediately with **HTTP 503 and `Retry-After`**, not queued. A rejection is a real HTTP status, so clients and load balancers can tell it from an answer.
**Tradeoff:** Under sudden traffic spikes, some requests are explicitly rejected. This is an intentional design decision: a predictable, bounded failure mode is safer and more observable than silent memory exhaustion or runaway billing.

## 9. In-Memory Query Embedding Cache
**Decision:** Cache recently computed query embeddings in a fixed-capacity, MD5-keyed in-memory dictionary.
**Rationale:** Many conversational RAG interactions involve follow-up queries that are semantically similar or even identical to a prior query. Re-embedding the same text via the hosted embedding API is a wasteful, latency-adding round-trip, and with Voyage's no-payment-method limit of 3 requests per minute it is also a scarce one. A 500-entry in-memory cache per embedding index, evicting the oldest entry first (FIFO: a hit does not refresh an entry) eliminates this for repeated queries at the cost of negligible RAM. The cache also remembers each query's token count, so an evaluation still charges every configuration what its embedding costs (Phase 6).
**Tradeoff:** Cache entries do not survive server restarts, and the cache is shared across all tenants (keyed purely on the query string hash). This is acceptable — query text itself is not sensitive, and cache misses simply fall back to a live API call with no correctness impact.

## 10. Admin-Provisioned Workspace Keys
**Decision:** There is no open registration endpoint. Tenant API keys are issued by an administrator through `POST /v1/admin/keys` and revoked through `POST /v1/admin/keys/revoke`, both authenticated with the admin key. There is no self-registration endpoint and no shared-tenant demo mode.
**Rationale:** Open sign-up makes rate limiting the only abuse control, and a shared-tenant mode collapses tenant isolation entirely. With admin-issued keys, every retrieval filters on `tenant_id` and returns nothing when the tenant is missing.
**Tradeoff:** Onboarding is an administrator action instead of self-service, and the admin key must be handled carefully. This suits a portfolio deployment with a small known set of users. A public product would need real sign-up (OAuth or invite tokens) in front of key issuance.

## 11. Ingest Source Validation
**Decision:** The `url` field of `POST /v1/documents` accepts only http(s) URLs whose host resolves exclusively to public addresses (`src/crawling/url_policy.py`).
**Rationale:** A URL field that reaches a fetcher unchecked is a way to make the server read internal addresses or local paths. The check is structural (scheme and resolved address, with a timeout on the lookup) and does not depend on site names.
**Tradeoff:** The address is validated when the request arrives and again by the fetch worker before each page, because DNS can change in between; a redirect target is checked too. Nothing we host fetches the page itself: a hosted reader service does, which is what makes DNS rebinding against our network moot.

## 12. LLM Access: One SDK Call Path, Explicit Retry, Fallback and Timeouts
**Decision:** LiteLLM is used as an SDK (`litellm.completion` / `litellm.acompletion`) for request formatting, response parsing and per-call cost lookup. Retry, fallback and timeouts are one explicit loop in `src/llm/client.py`, not `litellm.Router`.
**Rationale:** When this was built, Router's mid-stream fallback had open bugs (LiteLLM issues #28216 and #40404). Using the plain SDK for both streaming and non-streaming calls keeps one call path. Errors are classified by what a retry can achieve:
- *Transient* (429, 5xx, connection errors, and a 200 with an empty body, which Gemini's thinking mode can produce) are retried with exponential backoff, then fall back.
- *Permanent for this request* (404 dead model, 400, auth errors) are not retried locally, because an identical request cannot succeed; they go straight to the fallback.
- *Timeouts* also go straight to the fallback. Every call is bounded by `request_timeout_seconds` (60 s). LiteLLM's own default is 6,000 s, so without the bound a hung provider held a query slot for up to 100 minutes and the fallback never fired. Observed live on Gemini: a hung call reached the fallback after 218 s before the bound, and after 62 s with it.
- *Streams* fall back only before the first token, because a client that has received part of an answer cannot be handed a different model's answer.
**Tradeoff:** A legitimately slow answer longer than 60 s is cut off and answered by the fallback model instead. Proactive per-deployment rate pacing is not implemented; it is deferred to the configuration-search engine, where call volume requires it.

## 13. Durable Ingestion Queue: Procrastinate, and an API/Worker Split
**Decision:** `POST /v1/documents` no longer parses anything itself. It validates the request, writes the job (and an uploaded file's bytes) to Postgres, and defers a job onto [Procrastinate](https://procrastinate.readthedocs.io/), a job queue backed by the same database. A separate worker process claims jobs from that queue and does the actual fetching, parsing, chunking and embedding. The API and worker ship as two Docker images (`Dockerfile.api`, `Dockerfile.worker`) from the same source tree.
**Rationale:**
- *Why not in-process.* A FastAPI `BackgroundTasks` callback bounded by an in-process semaphore keeps its queue only in memory: a restart loses every queued or in-flight job, and ingestion capacity cannot scale apart from the web server.
- *Durability without new infrastructure.* Procrastinate's queue is plain Postgres tables, so it needed no new service (unlike Celery or RQ, which need Redis) and inherits the same durability, backups and connection story as everything else in Neon.
- *A real API/worker split, not just a queue.* Splitting into two processes is what makes ingestion capacity scale independently of query-serving capacity, and it is what lets the much heavier parser dependency (Docling) exist without bloating the API image. The API's own dependency list (`requirements-api.txt`) contains no document parser; a container leak check in CI (`docker run ... python -c "import docling"`, likewise torch and pymupdf) fails the build if that ever stops being true.
- *Atomicity.* A chunk write, a job-status update and a tenant token-usage update would be three separate statements, and a crash between them would leave the job status disagreeing with what was stored. `src/jobs/commit.py` makes them one transaction.
- *Locking in the queue, not in application code.* Two overlapping ingestion runs for the same document must not both write its chunks. Procrastinate's job-level lock (the document ID) makes the queue itself serialise them, which cannot be forgotten by a code path the way an application-level check could be.
**Recovery of orphaned jobs.** Procrastinate lists jobs whose worker stopped heart-beating but does not requeue them, so a periodic sweep (`src/jobs/recovery.py`) does: requeue while retry budget remains, otherwise fail the job in both the queue and the domain tables. The 30-second threshold is three times the worker's 10-second heartbeat interval, so a heartbeat missed to a Neon compute resume or a long GC pause is not mistaken for a dead worker; the cost is that an orphaned job waits at least that long, plus up to a minute for the next sweep. A worker that was merely paused, not dead, and comes back after its job was requeued is harmless: locks serialise the two runs, chunk writes are idempotent, and a rerun skips any job that already finished.
**Tradeoff:** Uploaded files travel through a Postgres table (`ingest_sources`) rather than a shared filesystem, which is the right call once the API and worker can run on different hosts, but it means Neon's storage now also holds transient upload bytes; a per-tenant cap (`MAX_PENDING_UPLOAD_BYTES`, 200 MB) bounds how much an offline or backlogged worker can accumulate. Two more processes to deploy and keep on the same database revision (both refuse to start on a schema mismatch).

## 14. Retention
**Decision:** The tables that grow with traffic are pruned by age (`src/maintenance.py`): `pipeline_events` after 14 days, `query_logs` and `fetch_log` after 90, failed queue rows after 30 days (successful ones are deleted as they finish). Pruning runs when a process starts (the API at boot, a worker at launch) and is idempotent.
**Rationale:** Neon's free plan holds 1 GB and the log tables grow with every query and job. The API host sleeps and the workers run on demand, so a timer would rarely be awake; running at startup uses the moments something is awake anyway. The windows are operational choices, not corpus-tuned: events are only useful while debugging recent behaviour, the query log is a dashboard history, and the fetch log is the audit trail that keeps abuse attributable (the daily quota needs only 24 hours of it).
**Tradeoff:** Pruning at startup means a long-running process that never restarts does not prune; at Tier 1 both processes restart often. Removing old rows loses history for anything older than the window: experiment results are not pruned.

## 15. The API Contract
**Decision:** Every resource is under `/v1` (chat, documents, jobs, workspace settings, usage, admin), and `openapi.json` at the repository root is generated from the app and checked by a test. Every non-2xx response has one body, `{code, message, request_id}`, and the same `request_id` is in the `X-Request-ID` header and the server log. A missing or wrong credential is always 401; the answer endpoints shed load with 503 and `Retry-After`; a model that cannot answer is 502 `generation_failed`; unexpected failures are 500 `internal_error` with no exception text.
**Rationale:** The Next.js client is generated from this document, so a change to a request or response shape must be a visible change to a committed file, caught by CI instead of by a user. A single error shape lets a client write one handler. A failure is never HTTP 200 with a sentence in the answer, which a client cannot tell from an answer, and a 500 never carries SQL or parameters.
**Tradeoff:** Paths and the admin header changed with no compatibility aliases, because nothing outside this repository consumed them. After the first external client, a breaking change means `/v2`.

## 16. One Chat Code Path, and Settings a Workspace Owns
**Decision:** The plain and the streaming answer endpoints share one service (`src/services/chat_service.py`): prepare (retrieve) first, then either `answer` or `events`. Retrieval runs before the streaming response starts, so a retrieval failure is an HTTP error and a failure after tokens have flowed is a typed `error` event. A workspace may store its own retrieval settings (`PUT /v1/workspace/settings`); chat runs the environment defaults, then the workspace's choices, then the request's `top_k` and reranker switch, through the same pipeline an experiment trial uses.
**Rationale:** Two copies of the same logic would have to be kept in step by hand: a fix to one (for example releasing the capacity slot) would have to be repeated in the other. Per-workspace settings are what let an experiment's winner be applied to chat without a deployment.
**Tradeoff:** Settings are validated as a whole `RetrievalConfig` before they are stored, so a combination that cannot run is refused when it is set, not when the next question arrives.

## 17. Observability: One Log Pipeline, Request Ids, Optional Error Reports
**Decision:** Every log line, from our code and from libraries, is one structured record written by one handler (`src/observability/logging_setup.py`, structlog's `ProcessorFormatter` over the standard `logging` module). It is JSON when stderr is not a terminal and readable text when it is (`LOG_FORMAT` forces either). The request middleware gives each request an id (the caller's `X-Request-ID` when well-formed), binds it to a context variable so every line written while serving the request carries it, including lines from threads started with `asyncio.to_thread`, echoes it in the response, and writes one access line per request: method, path, status and duration. Pipeline events (`src/observability/logger.py`) go through the same handler. Error reporting to Sentry is off unless `SENTRY_DSN` is set and sends exceptions only.
**Rationale:**
- *One handler.* The pipeline logger used to own a handler of its own while the root logger had another, so each event was printed twice. A single configuration, applied idempotently, cannot stack handlers; a test asserts one line per event.
- *Request ids in the log, not at call sites.* Binding the id once in the middleware means no function signature carries it, and a user-reported `request_id` (it is in every error body) finds every line of that request.
- *Questions and answers stay out of logs.* Logs are copied to third-party collectors. The log line for an event shows the length of a question, not its text, and the question and answer text of the generator and the rewriter are written only at debug level. The persisted `pipeline_events` row keeps the question text for debugging and is pruned after 14 days (section 14); `query_logs` keeps the query for the dashboard for 90.
- *Access lines never carry the query string*, which can hold tokens. Probes (`/health`, `/ready`) are logged at debug level because a platform polls them every few seconds.
- *Error reports carry no personal data.* PII collection is off, request bodies are never captured, local variables are not attached, and a `before_send` hook removes request headers (they hold the API key), cookies, body, query string and the user from every event. What is sent is the exception, its stack and the request id tag. Tracing is off. Sentry's FastAPI integration reports only server errors (5xx), so a 401 or a 429 is not an event.
**Tradeoff:** Structured logs are harder to read in a raw terminal, which is why a terminal gets the text renderer. Error reports and request ids cannot see a failure in a worker that never reaches Sentry's integration points; workers log through the same pipeline and report unhandled exceptions when `SENTRY_DSN` is set for them too.

## 18. Hosting the API: What Is and Is Not on Render
**Decision:** Only the API is hosted (`render.yaml`, free web service, Singapore, next to the Neon project). Workers run on demand on a laptop against the same database, so chat, stored results and the document list keep working while they are off. The platform health check is `/health`, which touches nothing; `/ready` (it queries the database) is for people and for a deployment that can afford to keep the database awake. Migrations are applied by hand before a deploy that adds one (`alembic upgrade head`, direct Neon endpoint), because a pre-deploy command is not available on the free plan, and the API refuses to start on a database that has not been migrated.
**Rationale:** A platform probes a health check every few seconds. If that check queried Neon, the compute would never reach its idle suspend and would spend its monthly compute hours doing nothing; `/health` still lets the platform replace an instance whose start-up failed, because it answers 503 in that case. `TRUSTED_PROXY_HOPS=1` is set because Render's load balancer is one hop in front of the API (section 6).
**Tradeoff:** A free web service sleeps after a period without traffic, so the first request after a pause waits for the instance to start and, if Neon was also idle, for its compute to resume (a few seconds each; both are Tier 1 costs the README states). The manual migration step is a way to forget something; the startup schema check turns forgetting into a clear refusal to start, not a failure at the first query.

### Neon compute hours: how to measure
Neon's free plan bounds compute time as well as storage, and the figure that matters is how many hours the compute was active, not how many requests were served. Read it in the Neon console (Monitoring, or the project's usage page) once a day for a week after deploying, with the workers off except when a document is ingested, and record each reading below. A reading that projects above the plan's monthly allowance means something is keeping the compute awake (a probe on `/ready`, a worker left running, a dashboard polling); find it before adding features.

| Date | Compute hours so far this month | Hours since the last reading | What ran |
|---|---|---|---|
| | | | |

## 19. Limits and Thresholds
Every limit the system enforces, where it lives, and why it has the value it has. *Fixed by a provider* means the value follows a documented or measured external limit; *operational* means a safety bound chosen to be generous for one person using the system, not tuned on a corpus; *experiment* means a starting point that retrieval or evaluation results have not validated.

| Limit | Value | Where | Kind and rationale |
|---|---|---|---|
| JSON request body | 256 KB | `src/api/app.py` | Operational: the largest valid chat request is a 2,000-character question plus 20 turns |
| Upload size | 20 MB (plus 1 MB framing) | `src/services/uploads.py` | Operational: bounds memory per upload; the body is read in chunks and refused past the limit |
| Chat question / history | 2,000 characters; 20 turns of 4,000 characters | `src/api/schemas/chat.py` | Operational: bounds what a caller can make us pay for in prompt tokens |
| `top_k` | 1 to 20 | `src/api/schemas/chat.py` | Operational; the HNSW `ef_search` of 100 covers the largest rerank pool (`top_k` x 4) |
| Rerank pool | 1 to 80 (chat: `top_k` x 4) | `src/retrieving/config.py`, `src/services/chat_config.py` | Experiment: depth against rerank latency |
| Context budget | 5,000 tokens | `src/generating/models.py` | Experiment: one request near 5.5K tokens fits Groq's free 8K tokens a minute |
| Answer requests | 5 a minute per key or address | `src/api/rate_limit.py` | Operational: each answer spends free-tier LLM quota |
| Ingest requests | 10 a minute | `src/api/rate_limit.py` | Operational |
| Read requests | 60 a minute | `src/api/rate_limit.py` | Operational |
| Admin requests | 10 a minute | `src/api/rate_limit.py` | Operational |
| Rejected credentials | 10 a minute per address, then 429 | `src/api/rate_limit.py` | Operational: slows key guessing without locking out a real user for long |
| Concurrent answers | 4 (`QUERY_CONCURRENCY`); excess is 503 with `Retry-After` | `src/config.py` | Operational: bounds memory and spend on a small host |
| Workspace chunks | 2,000 | `src/services/ingestion_service.py` | Operational: keeps one workspace within Neon's free storage |
| Jobs queued or running per workspace | 10 | `src/services/ingestion_service.py` | Operational: workers run on demand, so work can wait for days; this bounds the pile |
| Pending upload bytes per workspace | 200 MB | `src/services/ingestion_service.py` | Operational: leaves most of Neon's 1 GB for vectors |
| Web pages fetched per workspace per day | 200 (`FETCH_DAILY_PAGE_QUOTA`) | `src/config.py` | Fixed by a provider: protects the shared free reader allowances |
| Pages per sitemap job; child sitemaps | 50; 10 | `src/crawling/sitemap.py` | Operational: bounds one job's reader calls |
| Fetch pacing | one request per domain every 3 s | `src/config.py` | Fixed by a provider: keyless Jina allows about 20 requests a minute |
| PDF pages through Docling; parse timeout | 50; 900 s | `src/config.py` | Measured: worst rate about 8.5 s a page (Phase 1) |
| Unpacked DOCX size | 200 MiB | `src/parsing/child.py` | Experiment: 10 x the upload cap |
| Embedding requests (Voyage, no payment method) | 3 a minute, 10K tokens a minute | `src/config.py` | Fixed by a provider (measured 2026-10-02) |
| LLM call timeout | 60 s; 3 retries with 1, 2, 4 s backoff, then the fallback | `src/llm/client.py`, `src/llm/config.py` | Operational: a hung provider must not hold a slot for minutes |
| Database statement / lock timeout | 60 s / 10 s | `src/db/engine.py` | Operational: one stuck query must not hold a pooled connection |
| Database pool | 5 per engine (`DB_POOL_SIZE`) | `src/config.py` | Operational: Neon's free compute has few connections |
| Key validation cache | 60 s, 10,000 entries | `src/stores/api_keys.py` | Operational: bounds how long a revoked key can still work in another process |
| Query-embedding cache | 500 entries per index | `src/retrieving/dense.py` | Experiment |
| Retention | events 14 days; query log and fetch log 90; failed queue rows 30 | `src/maintenance.py` | Operational: Neon's 1 GB |
| Stalled-job threshold | 30 s of silence | `src/jobs/recovery.py` | Operational: three heartbeats, so a Neon resume is not read as a dead worker |
| Chunk sizes, overlap | 600 target, 800 max, 125 overlap, 150 min | Phase 3 | Experiment (table in Phase 3) |
| Cleaner scores and tiers | see Phase 2 | `src/processing/cleaner.py` | Experiment (table in Phase 2) |
| Regression gate | 90% valid runs; primary metric `mrr`; alpha 0.05 | Phase 6 | Operational: chosen in advance, not tuned |

Provider limits that shape the design, all from this project's own measurements or the providers' pages on the dates given in the phase documents, and all liable to change without notice: Voyage's card-free embedding and rerank limits (Phases 4 and 5), Groq's 8K tokens a minute and 200K a day, Gemini's free tier, measured on this project on 2026-10-06 at **20 requests a day per model** (and a few a minute), so chat on Gemini serves about 20 answers a day before the Groq fallback takes over, keyless Jina Reader's roughly 20 requests a minute, Firecrawl's 1,000 credits a month, and Neon's free storage and compute hours. Check each in its provider console before depending on it.
