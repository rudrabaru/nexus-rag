# Phase 9: Production Tradeoffs & Architecture Decisions

This document captures the explicit architectural tradeoffs and design decisions made to ensure the RAG system remains highly performant, resilient, and capable of operating as an efficient in-memory microservice.

## 1. API-First Model Architecture
**Decision:** Rely entirely on external APIs for both dense embedding generation and LLM generation, rather than loading local open-weight models.
**Rationale:** Loading modern LLMs and embedding models locally requires massive amounts of RAM and GPU resources. By delegating this compute to specialized external APIs, the core RAG microservice maintains a near-zero memory footprint for ML processing, allowing it to run on extremely resource-constrained infrastructure.
**Tradeoff:** The system introduces a hard dependency on external network calls. Network partitions or API outages will degrade functionality. To mitigate this, the system implements robust exponential backoff and retry logic, and supports seamless fallback providers.

## 2. Managed Cloud Vector Database
**Decision:** Utilize a managed cloud vector database (Qdrant Cloud) for embedding storage and similarity search rather than an in-process vector store.
**Rationale:** A managed cloud database simplifies deployment topologies, especially on ephemeral or stateless compute environments (like Render Free Tier, where Nexus RAG is currently deployed). It ensures persistence across container restarts and removes local disk dependency for vector storage.
**Tradeoff:** It introduces an external network hop on every query, adding network latency (typically 20–100ms) compared to an in-process database. Cost could also scale linearly with data volume depending on the provider.

## 3. In-Process Sparse Indexing
**Decision:** Utilize the native full-text search capabilities of an embedded relational database for keyword searching.
**Rationale:** This eliminates the need for a dedicated search cluster specifically for keyword indexing. Crucially, it supports continuous, incremental inserts as new documents are ingested, avoiding the expensive in-memory index rebuilds required by standard keyword search libraries.
**Tradeoff:** The tuning parameters of the embedded search engine are less configurable than standalone enterprise search engines. While highly effective for general domain text, extremely specialized corpora might require customized term-frequency weighting that is harder to achieve in this setup.

## 4. Concurrency & Queue Management
**Decision:** Enforce strict, semaphore-based concurrency limits on document ingestion.
**Rationale:** Ingestion involves memory-intensive tasks like extracting text from massive PDFs and rendering heavy web pages. Allowing unbounded concurrent ingestions would quickly lead to out-of-memory crashes on small servers.
**Tradeoff:** Users uploading multiple large documents simultaneously must wait in a queue. This UX impact is mitigated by a background job tracking system: clients receive an asynchronous tracking identifier immediately and can poll for status without blocking the main application thread.

## 5. Serverless Web Reading API vs. Local Headless Browser
**Decision:** Utilize an external serverless web reading API for web document crawling and conversion to clean markdown, rather than running a local headless browser or basic HTML scrapers.
**Rationale:** Modern web pages rely heavily on JavaScript (Single Page Applications) and complex DOM structures. Simple HTTP requests and HTML scrapers miss dynamically loaded content and generate noisy boilerplate (navbars, ads, footers). Running a local headless browser requires Chromium dependencies and consumes massive amounts of RAM per tab, which would immediately trigger out-of-memory crashes on resource-constrained environments like free tier hosting (where RAM limits are strictly enforced). Delegating DOM rendering, JS execution, and markdown cleaning to an external reading service maintains a near-zero local memory footprint.
**Tradeoff:** Introduces an external network dependency and rate-limiting constraints from a third-party service. While the reading engine handles anti-bot protection and JavaScript execution effectively, network latency or external service degradation can impact web ingestion latency. To mitigate this, the ingestion pipeline implements exponential backoff retries and structured error guards.

## 6. Rate Limiting Strategy
**Decision:** Implement application-layer rate limiting that respects reverse-proxy routing headers.
**Rationale:** In a production cloud environment, the application is almost always deployed behind a load balancer or ingress controller. Without reading forwarded client IP headers, the rate limiter would mistakenly throttle all traffic as coming from the single load balancer IP address.

## 7. Ephemeral Cloud Storage & Auto-Rehydration Strategy
**Decision:** Deploying the application on ephemeral hosting environments using a local embedded database and remote cloud vector database, paired with an automatic initialization routine upon server startup.
**Rationale:** This allows the entire infrastructure to be hosted on serverless or free-tier hosting without requiring expensive persistent disk storage. Vector payloads and metadata are persisted safely in the cloud vector database. On startup, an initialization routine downloads stored payloads and dynamically rebuilds the local full-text search index and document registries in memory.
**Tradeoff:** Ephemeral cloud instances spin down after inactivity and completely wipe the local filesystem. The system handles this through two complementary mechanisms:
1. **Stateless HMAC-Signed API Keys:** User credentials are cryptographically self-verifying. The key itself encodes the Tenant ID alongside its HMAC signature. Validation requires no database lookup, meaning user authentication survives ephemeral wipes unconditionally as long as the `RAG_API_KEY` environment secret remains constant across deployments.
2. **Automatic Document Registry Rebuild:** On each startup, the system compares chunk counts between Qdrant and the local SQLite registry. A divergence (Qdrant has chunks, SQLite is empty) is the canonical signal of an ephemeral wipe. The system responds automatically by scrolling all stored vector payloads from Qdrant and reconstructing the local SQLite registry and full-text search index, transparently restoring full search capability without user intervention.

A startup fail-fast guard enforces that the `RAG_API_KEY` environment secret is present before the server accepts any traffic. A misconfigured instance with a missing secret would silently accept registrations but produce HMAC signatures that are impossible to verify — the fail-fast prevents this class of silent misconfiguration.

## 8. Query Generation Concurrency Control
**Decision:** Apply a separate semaphore-based concurrency limit specifically to the query generation pipeline, distinct from the ingestion concurrency limit.
**Rationale:** LLM inference calls during query handling are memory-intensive and incur direct token-based API costs. Allowing unbounded simultaneous queries on a resource-constrained host risks OOM crashes and runaway costs. Excess requests beyond the cap are rejected immediately with an explicit "system busy" response rather than queued indefinitely, giving clients a deterministic signal to retry rather than waiting for an uncertain queue to drain.
**Tradeoff:** Under sudden traffic spikes, some requests are explicitly rejected. This is an intentional design decision: a predictable, bounded failure mode is safer and more observable than silent memory exhaustion or runaway billing.

## 9. In-Memory Query Embedding Cache
**Decision:** Cache recently computed query embeddings in a fixed-capacity, MD5-keyed in-memory dictionary.
**Rationale:** Many conversational RAG interactions involve follow-up queries that are semantically similar or even identical to a prior query. Re-embedding the same text via the external Jina API is a wasteful, latency-adding network round-trip. A 500-entry LRU-style in-memory cache eliminates this redundancy for repeated queries at the cost of negligible RAM.
**Tradeoff:** Cache entries do not survive server restarts, and the cache is shared across all tenants (keyed purely on the query string hash). This is acceptable — query text itself is not sensitive, and cache misses simply fall back to a live API call with no correctness impact.

## 10. Open Self-Registration Model
**Decision:** The workspace registration endpoint is open to any user without a pre-shared secret, with rate limiting as the sole abuse-prevention mechanism.
**Rationale:** A pre-shared registration secret requires synchronizing that secret across every separately deployed service that hosts the frontend UI (such as Streamlit Cloud). In a multi-platform deployment architecture, this synchronization is fragile: a mismatch between secrets on the UI server and the API server silently produces 403 errors for every legitimate user attempting to sign up. The registration endpoint also provides no meaningful server-side privilege — it only creates an isolated private workspace — so gating it with a shared secret provides negligible security while introducing significant operational complexity.
**Tradeoff:** Without a secret gate, any actor who discovers the API endpoint can create workspaces. The rate limiter bounds the practical impact of this. In a future version where workspace creation has a real cost (e.g., provisioning external resources per tenant), a proper authentication mechanism (OAuth, invite tokens) would be warranted. For the current architecture, open registration is the correct tradeoff.
