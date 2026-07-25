# Phase 9: Production Tradeoffs & Architecture Decisions

This document captures the explicit architectural tradeoffs and design decisions made to ensure the RAG system remains highly performant, resilient, and capable of operating as an efficient in-memory microservice.

## 1. API-First Model Architecture
**Decision:** Rely entirely on external APIs for both dense embedding generation and LLM generation, rather than loading local open-weight models.
**Rationale:** Loading modern LLMs and embedding models locally requires massive amounts of RAM and GPU resources. By delegating this compute to specialized external APIs, the core RAG microservice maintains a near-zero memory footprint for ML processing, allowing it to run on extremely resource-constrained infrastructure.
**Tradeoff:** The system introduces a hard dependency on external network calls. Network partitions or API outages will degrade functionality. To mitigate this, the system implements robust exponential backoff and retry logic, and supports seamless fallback providers.

## 2. In-Process Vector Database
**Decision:** Embed the vector database directly into the application process using local persistent storage, rather than deploying a standalone vector database cluster (like Milvus or Qdrant).
**Rationale:** An in-process database significantly simplifies the deployment topology and operational overhead. It provides sub-millisecond approximate nearest-neighbor search without the latency of an extra network hop.
**Tradeoff:** Local disk I/O can become a bottleneck on highly constrained instances. Furthermore, if the application scales horizontally to multiple containers, they will not natively share the same vector index without an external shared volume or migrating to a managed cloud vector database.

## 3. In-Process Sparse Indexing
**Decision:** Utilize the native full-text search capabilities of an embedded relational database (SQLite FTS5) for keyword searching.
**Rationale:** This eliminates the need for external dependencies or dedicated search clusters (like Elasticsearch). Crucially, it supports continuous, incremental inserts as new documents are ingested, avoiding the expensive in-memory index rebuilds required by standard Python BM25 libraries.
**Tradeoff:** The tuning parameters of the embedded search engine are less configurable than standalone implementations. While highly effective for general domain text, extremely specialized corpora might require customized term-frequency weighting that is harder to achieve in this setup.

## 4. Concurrency & Queue Management
**Decision:** Enforce strict, semaphore-based concurrency limits on document ingestion.
**Rationale:** Ingestion involves memory-intensive tasks like extracting text from massive PDFs and rendering heavy web pages. Allowing unbounded concurrent ingestions would quickly lead to out-of-memory (OOM) crashes on small servers.
**Tradeoff:** Users uploading multiple large documents simultaneously must wait in a queue. This UX impact is mitigated by a background job tracking system: clients receive a Job ID immediately and can poll for status asynchronously without blocking the main application thread.

## 5. Headless Browser Crawling
**Decision:** Use a headless browser to render HTML pages during web ingestion, rather than simple HTTP GET requests.
**Rationale:** Modern web pages heavily rely on JavaScript (Single Page Applications) to load content dynamically. Simple scrapers miss this content entirely.
**Tradeoff:** Headless browsers have a massive RAM footprint. Rendering complex pages adds significant overhead to the ingestion pipeline. Furthermore, sites utilizing aggressive bot-protection services may actively block headless browser engines.

## 6. Rate Limiting Strategy
**Decision:** Implement application-layer rate limiting that respects reverse-proxy headers (like `X-Forwarded-For`).
**Rationale:** In a production cloud environment, the application is almost always deployed behind a load balancer or ingress controller. Without reading the forwarded IP headers, the rate limiter would mistakenly throttle all traffic as coming from the single load balancer IP.
