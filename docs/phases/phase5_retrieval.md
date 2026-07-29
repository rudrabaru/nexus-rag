# Phase 5: Retrieval

## Overview
The retrieval phase surfaces the most relevant chunks from the indexed knowledge base for a given query. The system employs a sophisticated three-stage architecture: Hybrid Search (combining Semantic and Keyword matching), Reciprocal Rank Fusion, and optional Cross-Encoder Reranking. All stages are measurable and independently observable.

## Core Implementation Logic

### Multi-Tenancy & Strict Security Filtering
Before any chunk is evaluated for relevance, a strict security filter is enforced directly at the storage and retrieval layer:
- In production execution, if a tenant identifier is missing, unassigned, or set to a wildcard, the query is immediately rejected and returns an empty result set without querying the underlying databases.
- When a valid tenant identifier is provided, queries are strictly scoped using database-level payload filters and exact query predicates.
- For offline evaluation and benchmarking pipelines, an explicit, trusted administrative override allows cross-tenant evaluation without risking production leakage.

### Standalone Retrieval Modes for scientific Ablation
The architecture decouples retrieval modes into modular components to support scientific ablation and benchmarking:
- **Dense Search Mode:** Relies purely on vector embeddings and cosine similarity to capture conceptual meaning.
- **Sparse Search Mode:** Wraps an embedded full-text search index with keyword scoring and linguistic stemming.
- **Hybrid Fusion Mode:** Executes dense and sparse search concurrently and fuses their ranks via Reciprocal Rank Fusion.

### Stage 1: Dense Retrieval (Semantic Search)
1. The user's query is embedded via an external service, specifically tagged with a query-specific task profile to optimize for searching.
2. The vector database is queried to find the top candidates based on pure mathematical similarity (cosine distance).
3. These results capture the *meaning* and *concepts* of the query, even if the exact words don't match.

### Stage 2: Sparse Retrieval (Keyword Search)
In parallel, the local full-text search database is queried with the raw string. This engine applies linguistic stemming and a keyword scoring algorithm to find exact terminology matches. This is critical for queries involving specific acronyms, error codes, or proper nouns that semantic search might misinterpret.

### Stage 3: Reciprocal Rank Fusion (RRF)
The semantic and keyword results are completely different mathematically and cannot be simply added together. The system fuses them using **Reciprocal Rank Fusion (RRF)**.

RRF looks at the *rank order* of the results rather than their raw scores. A document that appears high in both the semantic list and the keyword list will be boosted to the absolute top of the final fused list. This mathematical approach guarantees the best of both worlds without requiring brittle, manual score weighting.

### Stage 4: Optional Cross-Encoder Reranking
If configured, the top results from the fused list are sent to a specialized external cross-encoder reranking service. 
Unlike standard embeddings that look at the query and the document in isolation, a cross-encoder reads the query and the document *together* at the same time, producing a highly calibrated relevance score that accounts for their joint context. 

This stage replaces the rank-fused order with the reranker's precise relevance scores. If the reranker is disabled via environment configuration, the system gracefully falls back to the rank-fused ordering.

### Score Calibration
Relevance scores are treated as **ranking signals, not absolute cutoffs**. The system defaults to maximizing recall by allowing all retrieved top candidates through, rather than applying an arbitrary minimum score cutoff that might accidentally filter out the correct answer.

### Latency Observability
The retrieval pipeline is highly instrumented, reporting discrete timing for each micro-stage (embedding the query, searching the databases, reranking). This allows operators to easily identify performance bottlenecks in production.

## Design Philosophy & Tradeoffs
- **Reranker Latency vs. Accuracy:** The cross-encoder reranker significantly boosts the accuracy of the top results but requires an additional API call, adding latency to the overall query time. For speed-critical applications, it can be disabled with a slight penalty to absolute precision.
