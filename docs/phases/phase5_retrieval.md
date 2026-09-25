# Phase 5: Retrieval

## Overview
The retrieval phase surfaces the most relevant chunks from the indexed knowledge base for a given query. The system employs a sophisticated three-stage architecture: Hybrid Search (combining Semantic and Keyword matching), Reciprocal Rank Fusion, and optional Cross-Encoder Reranking. All stages are measurable and independently observable.

## Core Implementation Logic

### Multi-Tenancy & Strict Security Filtering
Before any chunk is evaluated for relevance, a strict security filter is enforced directly at the storage and retrieval layer:
- In production execution, if a tenant identifier is missing, unassigned, or set to a wildcard, the query is immediately rejected and returns an empty result set without querying the underlying databases.
- When a valid tenant identifier is provided, both searches carry a `tenant_id = :tenant` SQL predicate on the one `chunks` table.
- Chunks are keyed by `(tenant_id, chunk_id)`. The previous vector store keyed them by `chunk_id` alone, and chunk IDs derive from the source URL, so a second tenant ingesting the same page overwrote the first tenant's vectors. The legacy data showed it: all 260 chunks of one page ingested by two tenants were held under the second tenant only.
- For offline evaluation and benchmarking pipelines, an explicit, trusted administrative override allows cross-tenant evaluation without risking production leakage.

### Standalone Retrieval Modes for scientific Ablation
The architecture decouples retrieval modes into modular components to support scientific ablation and benchmarking:
- **Dense Search Mode:** Relies purely on vector embeddings and cosine similarity to capture conceptual meaning.
- **Sparse Search Mode:** Wraps an embedded full-text search index with keyword scoring and linguistic stemming.
- **Hybrid Fusion Mode:** Executes dense and sparse search concurrently and fuses their ranks via Reciprocal Rank Fusion.

### Stage 1: Dense Retrieval (Semantic Search)
1. The user's query is embedded via an external service, specifically tagged with a query-specific task profile to optimize for searching.
2. Postgres returns the nearest chunks by cosine distance (`ORDER BY embedding <=> :query LIMIT k`) through a pgvector HNSW index over `halfvec(1024)` columns.
3. These results capture the *meaning* and *concepts* of the query, even if the exact words don't match.

**Filtered approximate search.** An HNSW index scan finds a fixed-size candidate list (`hnsw.ef_search`) and the tenant filter is applied afterwards, so a tenant owning a small share of the index could get fewer than `k` results. pgvector 0.8 added iterative scans (`hnsw.iterative_scan = relaxed_order`), which keep scanning until `k` rows pass the filter. `ef_search = 100` covers the largest candidate pool the API can request (rerank pool = `top_k × 4`, `top_k ≤ 20`). `relaxed_order` can return rows slightly out of order, so the store re-sorts by score. Both settings are applied when a connection opens, not per query.

### Stage 2: Sparse Retrieval (Keyword Search)
Concurrently, Postgres full-text search runs over a generated `tsvector` column (`to_tsvector('english', chunk_text)`, GIN-indexed). The query goes through `plainto_tsquery`, which treats every character as plain text, so query syntax cannot be injected. All terms must match first (AND). When nothing matches, the same terms are retried with any-term matching (OR), and that fallback is logged. Results are ranked by `ts_rank_cd`, which is not BM25. RRF consumes only the rank order, so only the ordering matters.

Because the sparse index is a generated column of the same row as the vector, it cannot fall out of sync with it. The SQLite FTS5 index it replaces held 1,162 rows against 2,284 vectors, so hybrid search had been searching about half the corpus for keywords.

Known limitation: `'english'` stemming is applied to every document. A non-English corpus needs a per-document text-search configuration (language detection already exists in the ingestion stage).

### Stage 3: Reciprocal Rank Fusion (RRF)
The semantic and keyword results are completely different mathematically and cannot be simply added together. The system fuses them using **Reciprocal Rank Fusion (RRF)**.

RRF looks at the *rank order* of the results rather than their raw scores. A document that appears high in both the semantic list and the keyword list will be boosted to the absolute top of the final fused list. This mathematical approach guarantees the best of both worlds without requiring brittle, manual score weighting.

RRF fuses on `chunk_id`, so both lists must use the same identifier. Until the Postgres migration they did not: dense results carried the vector store's UUID point ID while sparse results carried the real chunk ID. A chunk found by both retrievers therefore never received a fused score, and it could occupy two of the five result slots. Measured on the benchmark's 38 queries: 27 of 190 top-5 slots (14.2%) held a duplicate, so the generator saw four distinct chunks instead of five in 27 queries. Recall did not change (see Phase 6 for why the benchmark could not detect it).

**Filtered-search settings on Neon.** The HNSW settings above must reach every search connection. Neon's proxy silently drops individual startup parameters (tested with `work_mem`, which was ignored) but forwards libpq's `options` parameter, so the settings are sent as `options=-chnsw.iterative_scan=relaxed_order -chnsw.ef_search=100`. An integration test asserts the values actually arrive in the session. Without it, filtered search would have silently lost iterative scans.

### Stage 4: Optional Cross-Encoder Reranking
If configured, the top results from the fused list are sent to a specialized external cross-encoder reranking service. 
Unlike standard embeddings that look at the query and the document in isolation, a cross-encoder reads the query and the document *together* at the same time, producing a highly calibrated relevance score that accounts for their joint context. 

This stage replaces the rank-fused order with the reranker's precise relevance scores. If the reranker is disabled via environment configuration, the system gracefully falls back to the rank-fused ordering.

### Score Calibration
Relevance scores are treated as **ranking signals, not absolute cutoffs**. The system defaults to maximizing recall by allowing all retrieved top candidates through, rather than applying an arbitrary minimum score cutoff that might accidentally filter out the correct answer.

### Latency Observability
The retrieval pipeline is highly instrumented, reporting discrete timing for each micro-stage (embedding the query, searching the databases, reranking). This allows operators to easily identify performance bottlenecks in production.

## Design Philosophy & Tradeoffs
- **Reranker: Latency AND Accuracy Tradeoff:** Contrary to the common assumption that adding a cross-encoder reranker always improves results, empirical ablation against our multi-domain benchmark revealed a concrete regression: the **Jina AI Reranker** (`jina-reranker-v2-base-multilingual`) lowered Recall@1 from **0.974 (Hybrid alone) to 0.816**, while maintaining 1.000 Recall@5. The root cause was the off-the-shelf reranker over-indexing on general summary overview chunks rather than specific technical sub-sections. The reranker is therefore exposed as an **optional runtime toggle** — recommended only for non-interactive, latency-tolerant tasks where deeper cross-attention across the top-5 candidates is more valuable than pinpoint top-1 precision.
