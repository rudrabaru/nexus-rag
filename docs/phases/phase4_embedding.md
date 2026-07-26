# Phase 4: Embedding

## Overview
The embedding phase transforms textual chunks into mathematical representations, enabling similarity search algorithms to locate relevant context based on user queries. The system uses a unified, API-first embedding approach for both dense (semantic) vectors and sparse (keyword) indexing.

## Core Implementation Logic

### Dense Embeddings
The system delegates all dense vector generation to a highly optimized external embedding API (specifically, Jina's `jina-embeddings-v3`).

- **Asymmetric Encoding:** The system uses task-aware encoding. Chunks processed during ingestion are encoded with the `retrieval.passage` task type, optimizing them to be retrieved. User queries are encoded with the `retrieval.query` task type, optimizing them for searching. This asymmetric approach is critical for high-fidelity late-interaction models.
- **Batching & Concurrency:** Chunks are grouped into specific batches of 50 and sent in parallel to the Jina API to maximize throughput without exceeding payload limits.
- **Resiliency:** The system employs exponential backoff and retry logic (up to 3 retries) to gracefully absorb transient network failures or API rate limits.
- **Zero RAM Footprint:** No local embedding model is loaded into memory. All dense embedding computation is remote. This is a deliberate architectural tradeoff that frees significant RAM for the vector database, headless browser, and the web server, allowing the entire system to run comfortably on resource-constrained micro-instances.

### Sparse Indexing
Sparse keyword search is implemented via an embedded, high-performance full-text search database engine.

- During ingestion, every chunk's text is inserted into a local virtual table alongside its metadata.
- The index applies stemming algorithms natively (e.g., matching "running" with "run") to handle morphological variations.
- Results are scored using industry-standard BM25 algorithms built directly into the database engine.
- **Incremental Updates:** Unlike traditional Python-based BM25 libraries that require a full in-memory rebuild of the index after every document is added, this embedded database approach supports continuous, incremental inserts. This makes it highly resilient and suited for a long-running microservice.

### Metadata Injection
Every generated embedding is permanently tagged and stored in the vector database with rich metadata:
- **Tenant ID and Visibility:** Ensures strict security filtering at the database level.
- **Source Document and URL:** Provides the bedrock for accurate citations.
- **Heading Path:** Injected as a structured list, allowing the generation phase to cite exact section-level hierarchy.
- **Chunk Type:** Identifies whether the chunk is text, code, table, or mixed, enabling observability and potential type-specific retrieval boosting.

### Chunk Size Enforcement
Before embedding, the system enforces a hard token limit on every chunk to ensure it safely fits within the maximum context window of the external embedding API, preventing outright ingestion failures due to oversized blocks.

## Design Philosophy & Tradeoffs
- **Network Dependency:** All dense embedding generation requires outbound API calls. A network partition will cause ingestion to fail gracefully (with retries), but there is no local fallback embedding model.
- **API Rate Limits:** Heavy, sustained ingestion loads may encounter third-party API rate limits. The exponential backoff strategy mitigates this, but massive batch uploads are intentionally paced to respect external provider limits.
