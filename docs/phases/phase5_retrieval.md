# Phase 5: Retrieval

## Overview
The retrieval phase surfaces the most relevant chunks from the indexed knowledge base for a given query. The system employs a sophisticated three-stage architecture: Hybrid Search (combining Semantic and Keyword matching), Reciprocal Rank Fusion, and optional Cross-Encoder Reranking. All stages are measurable and independently observable.

## Core Implementation Logic

### Multi-Tenancy & Security Filtering
Before any chunk is evaluated for relevance, a strict security filter is enforced directly at the vector database level.
- If a chunk is marked "public", it is available to all queries.
- If a chunk is marked "private", it is only returned if the querying user's Tenant ID matches the chunk's Tenant ID.

This filter is applied at the database query level — not as an after-the-fact post-filter. This guarantees that performance and accuracy are not impacted by the total size of the corpus, and ensures absolute data isolation.

### Stage 1: Dense Retrieval (Semantic Search)
1. The user's query is embedded via the external API, specifically tagged with a "query" task type to optimize for searching.
2. The vector database is queried to find the top candidates based on pure mathematical similarity (cosine distance).
3. These results capture the *meaning* and *concepts* of the query, even if the exact words don't match.

### Stage 2: Sparse Retrieval (Keyword Search)
In parallel, the local full-text search database is queried with the raw string. This engine applies linguistic stemming and a BM25 scoring algorithm to find exact keyword matches. This is critical for queries involving specific acronyms, error codes, or proper nouns that semantic search might misinterpret.

### Stage 3: Reciprocal Rank Fusion (RRF)
The semantic and keyword results are completely different mathematically and cannot be simply added together. The system fuses them using **Reciprocal Rank Fusion (RRF)**.

RRF looks at the *rank order* of the results rather than their raw scores. A document that appears high in both the semantic list and the keyword list will be boosted to the absolute top of the final fused list. This mathematical approach guarantees the best of both worlds without requiring brittle, manual score weighting.

### Stage 4: Optional Cross-Encoder Reranking
If configured, the top results from the fused list are sent to a specialized Cross-Encoder Reranking API (e.g., Jina Reranker). 
Unlike standard embeddings that look at the query and the document in isolation, a cross-encoder reads the query and the document *together* at the same time, producing a highly calibrated relevance score that accounts for their joint context. 

This stage replaces the RRF scores with the reranker's precise relevance scores. If the reranker is disabled via environment configuration, the system gracefully falls back to the RRF ranking.

### Score Calibration
Relevance scores are treated as **ranking signals, not absolute cutoffs**. The system defaults to maximizing recall by allowing all retrieved top candidates through, rather than applying an arbitrary minimum score cutoff that might accidentally filter out the correct answer.

### Latency Observability
The retrieval pipeline is highly instrumented, reporting discrete timing for each micro-stage (embedding the query, searching the databases, reranking). This allows operators to easily identify performance bottlenecks in production.

## Design Philosophy & Tradeoffs
- **Reranker Latency vs. Accuracy:** The cross-encoder reranker significantly boosts the accuracy of the top results but requires an additional API call, adding latency to the overall query time. For speed-critical applications, it can be disabled with a slight penalty to absolute precision.
