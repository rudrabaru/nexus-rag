# Phase 4: Embedding

## Overview
The embedding phase transforms textual chunks into mathematical representations, enabling similarity search algorithms to locate relevant context based on user queries. The system uses a unified, API-first embedding approach for both dense (semantic) vectors and sparse (keyword) indexing.

## Core Implementation Logic

### Dense Embeddings
The system delegates all dense vector generation to the **Jina Embeddings v3** API (`jina-embeddings-v3`), a highly optimized external multilingual embedding service.

- **Asymmetric Encoding:** The system uses task-aware encoding. Chunks processed during ingestion are encoded with a passage-specific task profile, optimizing them to be retrieved. User queries are encoded with a query-specific task profile, optimizing them for searching. This asymmetric approach is critical for high-fidelity late-interaction models.
- **Batching & Concurrency:** Chunks are grouped into specific batches of 50 and sent in parallel to the external embedding service to maximize throughput without exceeding payload limits.
- **Resiliency & Partial Success:** The system employs exponential backoff and retry logic (up to 3 retries) to absorb transient network failures or API rate limits. If rate limits persist after retries, the pipeline isolates the failed chunk batches without aborting the entire document. Successfully embedded chunks are committed to Postgres, while the job status transitions to partial success and records an explicit diagnostic error reason in the document metadata.
- **Zero RAM Footprint:** No local embedding model is loaded into memory. All dense embedding computation is remote. This is a deliberate architectural tradeoff that frees significant RAM for the web server and document processing pipeline, allowing the entire system to run comfortably on resource-constrained micro-instances.
- **Query Embedding Cache:** To avoid redundant external API calls for repeated or near-identical queries, a fixed-capacity (500-entry) MD5-keyed in-memory cache stores recently computed query embeddings. Cache hits serve queries entirely from memory at near-zero latency.

### Storage: one row per chunk in Postgres

Each embedded chunk is written as one row of the `chunks` table in Postgres (Neon), holding its text, its vector (`halfvec(1024)`, HNSW-indexed) and a generated full-text column (`tsvector`, GIN-indexed).

- **No separate sparse write.** The keyword index is computed by Postgres from the same row, so the dense and sparse indexes cannot diverge. The previous design wrote vectors to Qdrant and text to a local SQLite FTS5 table in two separate writes, and on the live deployment they had diverged (2,284 vectors, 1,162 keyword rows).
- **Idempotent upserts** keyed by `(tenant_id, chunk_id)`: re-ingesting a document replaces its rows in place.
- **Cascading deletes:** chunks reference their document, so deleting a document removes its chunks, vectors and keyword entries in one transaction.
- **`halfvec` (float16) storage** halves vector size (about 2 KB per chunk), which matters under Neon's 0.5 GB free tier (roughly 100k+ chunks after index overhead). On unit-length 1024-dimensional vectors the precision loss leaves the cosine between stored and original vector above 0.9999, verified during migration.
- **Provenance:** every row records the `embedding_model` that produced it. Vectors from different models live in different spaces and must never be compared; making the model a per-index choice is item 8.

### Metadata Injection
Every generated embedding is stored with rich metadata:
- **Tenant ID and Visibility:** Ensures strict security filtering at the database level.
- **Source Document and URL:** Provides the bedrock for accurate citations.
- **Heading Path:** Injected as a structured list, allowing the generation phase to cite exact section-level hierarchy.
- **Chunk Type:** Identifies whether the chunk is text, code, table, or mixed, enabling observability and potential type-specific retrieval boosting.

### Chunk Size Enforcement
Before embedding, the system enforces a hard token limit on every chunk to ensure it safely fits within the maximum context window of the external embedding API, preventing outright ingestion failures due to oversized blocks.

## Design Philosophy & Tradeoffs
- **Network Dependency:** All dense embedding generation requires outbound API calls. A network partition will cause ingestion to fail gracefully (with retries), but there is no local fallback embedding model.
- **API Rate Limits vs. Partial Indexing:** Heavy, sustained ingestion loads may encounter third-party API rate limits. Rather than failing an entire document when an embedding batch is rate-limited, the system opts for partial indexing. This preserves all successfully embedded chunks for immediate retrieval while surfacing diagnostic error reasons to the user for observability.
