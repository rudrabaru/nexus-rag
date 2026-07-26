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
**Decision:** Utilize the native full-text search capabilities of an embedded relational database (SQLite FTS5) for keyword searching.
**Rationale:** This eliminates the need for a dedicated search cluster (like Elasticsearch) specifically for keyword indexing. Crucially, it supports continuous, incremental inserts as new documents are ingested, avoiding the expensive in-memory index rebuilds required by standard Python BM25 libraries.
**Tradeoff:** The tuning parameters of the embedded search engine are less configurable than standalone implementations. While highly effective for general domain text, extremely specialized corpora might require customized term-frequency weighting that is harder to achieve in this setup.

## 4. Concurrency & Queue Management
**Decision:** Enforce strict, semaphore-based concurrency limits on document ingestion.
**Rationale:** Ingestion involves memory-intensive tasks like extracting text from massive PDFs and rendering heavy web pages. Allowing unbounded concurrent ingestions would quickly lead to out-of-memory (OOM) crashes on small servers.
**Tradeoff:** Users uploading multiple large documents simultaneously must wait in a queue. This UX impact is mitigated by a background job tracking system: clients receive a Job ID immediately and can poll for status asynchronously without blocking the main application thread.

## 5. Web Crawling via Jina Reader API vs. Local Headless Browser
**Decision:** Utilize the external Jina Reader API for web document crawling and conversion to clean markdown, rather than running a local headless browser (e.g., Playwright or Puppeteer) or basic HTML scrapers (e.g., BeautifulSoup).
**Rationale:** Modern web pages rely heavily on JavaScript (Single Page Applications) and complex DOM structures. Simple GET requests and HTML scrapers miss dynamically loaded content and generate noisy boilerplate (navbars, ads, footers). Running a local headless browser requires Chromium dependencies and consumes massive amounts of RAM (100–300MB+ per tab), which would immediately trigger out-of-memory (OOM) crashes on resource-constrained environments like Render's Free Tier (512MB total RAM limit). Delegating DOM rendering, JS execution, and markdown cleaning to Jina Reader maintains a near-zero local memory footprint.
**Tradeoff:** Introduces an external network dependency and rate-limiting constraints from a third-party service. While Jina Reader handles anti-bot protection and JavaScript execution effectively, network latency or external service degradation can impact web ingestion latency. To mitigate this, the ingestion pipeline implements exponential backoff retries and structured error guards.

## 6. Rate Limiting Strategy
**Decision:** Implement application-layer rate limiting that respects reverse-proxy headers (like `X-Forwarded-For`).
**Rationale:** In a production cloud environment, the application is almost always deployed behind a load balancer or ingress controller. Without reading the forwarded IP headers, the rate limiter would mistakenly throttle all traffic as coming from the single load balancer IP.

## 7. Render Free Tier & Ephemeral File System (The "15-Minute Wipe" Issue)
**Decision:** Deploying the application on Render's Free Tier using a local SQLite database and remote Qdrant database, with an auto-rebuild script on startup.
**Rationale:** This allows the entire infrastructure to be hosted for $0/month. The vector data is persisted safely in Qdrant. On startup, `startup.py` downloads the Qdrant payloads and dynamically rebuilds the SQLite FTS index and document registry.
**Tradeoff:** Render's Free Tier spins down after 15 minutes of inactivity and completely wipes the local filesystem. While the documents and chunks are successfully rebuilt from Qdrant, the `api_keys` table (which maps the raw API key to the hashed `tenant_id`) is permanently lost. This results in users receiving `403 Forbidden` errors if they attempt to use their original API keys after the server restarts. This is a deliberate tradeoff for free hosting; production deployments must mount a Persistent Disk ($0.25/month) or migrate to a managed relational database to permanently store API keys.
