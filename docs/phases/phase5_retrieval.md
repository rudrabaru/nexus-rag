# Phase 5: Retrieval

## Overview
The retrieval phase surfaces the most relevant chunks from the indexed knowledge base for a given query. Retrieval runs as a pipeline assembled from one explicit configuration: a first stage (dense, sparse, or both fused by weighted Reciprocal Rank Fusion) and an optional cross-encoder rerank of the first-stage pool. Every stage is measurable and independently observable.

## Core Implementation Logic

### Multi-Tenancy & Strict Security Filtering
Before any chunk is evaluated for relevance, a strict security filter is enforced directly at the storage and retrieval layer:
- In production execution, if a tenant identifier is missing, unassigned, or set to a wildcard, the query is immediately rejected and returns an empty result set without querying the underlying databases.
- When a valid tenant identifier is provided, both searches carry a `tenant_id = :tenant` SQL predicate on the one `chunks` table.
- Chunks are keyed by `(tenant_id, index_id, chunk_id)`. The previous vector store keyed them by `chunk_id` alone, and chunk IDs derive from the source URL, so a second tenant ingesting the same page overwrote the first tenant's vectors. The legacy data showed it: all 260 chunks of one page ingested by two tenants were held under the second tenant only.
- For offline evaluation and benchmarking pipelines, an explicit, trusted administrative override allows cross-tenant evaluation without risking production leakage.

### One Configuration, One Pipeline (`src/retrieving/config.py`, `pipeline.py`)
Every query-time retrieval knob lives in one `RetrievalConfig`:

| Knob | Default | Meaning |
|---|---|---|
| `strategy` | `hybrid` | `dense`, `sparse`, or both fused |
| `top_k` | 5 | results returned |
| `rrf_k` | 60 | RRF constant (Cormack et al. 2009's value, not tuned here) |
| `dense_weight`, `sparse_weight` | 1.0, 1.0 | weight of each ranking in the fusion; 0 skips that search |
| `reranker` | none | `flashrank`, `jina` or `voyage` |
| `rerank_candidates` | 20 | first-stage pool the reranker reorders (≤ 80, see ef_search below) |
| `fusion_depth` | none | hybrid only: how many results dense and sparse each return before fusion. Fusing two lists of `top_k` cannot surface a chunk that one list ranked `top_k + 1`; a deeper first stage lets agreement between the lists promote it. Must be at least the candidates it feeds. An experiment knob; chat does not set it |
| `index_id` | configured index | which embedding index to search |

`build_pipeline(config, resources)` assembles a pipeline from process-wide resources that are built once: per-index retrievers (with their query-embedding cache) and loaded rerankers. Pipelines are cheap, so **chat builds one per request** from the default configuration (`RETRIEVAL_STRATEGY`, `RERANKER`) plus the request's `top_k` and reranker toggle, and **an evaluation builds one per configuration** (each trial of an experiment spec, Phase 6). There is no retriever fixed at startup any more, so a configuration measured offline is exactly the one chat serves.

**Degradation is explicit.** If a hybrid search cannot embed the query (an embedding outage or rate limit), it serves the sparse ranking; if a reranker fails, the first-stage order is kept. Either way the reason is recorded on the result (`degraded`), logged, and counted in evaluation reports (`degraded_queries`), so an evaluation never silently measures something other than the configuration it names. A database failure is not degradation and raises.

### Stage 1: Dense Retrieval (Semantic Search)
1. The user's query is embedded via an external service, specifically tagged with a query-specific task profile to optimize for searching.
2. Postgres returns the nearest chunks by cosine distance (`ORDER BY embedding <=> :query LIMIT k`) through a pgvector HNSW index over `halfvec(1024)` columns.
3. These results capture the *meaning* and *concepts* of the query, even if the exact words don't match.

**Filtered approximate search.** An HNSW index scan finds a fixed-size candidate list (`hnsw.ef_search`) and the tenant filter is applied afterwards, so a tenant owning a small share of the index could get fewer than `k` results. pgvector 0.8 added iterative scans (`hnsw.iterative_scan = relaxed_order`), which keep scanning until `k` rows pass the filter. `ef_search = 100` covers the largest candidate pool the API can request (rerank pool = `top_k × 4`, `top_k ≤ 20`). `relaxed_order` can return rows slightly out of order, so the store re-sorts by score. Both settings are applied when a connection opens, not per query.

### Stage 2: Sparse Retrieval (Keyword Search)
Concurrently, Postgres full-text search runs over a generated `tsvector` column (`to_tsvector('english', chunk_text)`, GIN-indexed). The query goes through `plainto_tsquery`, which treats every character as plain text, so query syntax cannot be injected. All terms must match first (AND). When nothing matches, the same terms are retried with any-term matching (OR), and that fallback is logged. Results are ranked by `ts_rank_cd`, which is not BM25. RRF consumes only the rank order, so only the ordering matters.

Dense and sparse results are ordered by score and then by chunk id: the keyword rank (`ts_rank_cd`) ties constantly and identical-text chunks tie on distance, and an unordered tie made a rerun rank differently and added noise to every significance test. Because the sparse index is a generated column of the same row as the vector, it cannot fall out of sync with it. The SQLite FTS5 index it replaces held 1,162 rows against 2,284 vectors, so hybrid search had been searching about half the corpus for keywords.

Known limitation: `'english'` stemming is applied to every document. A non-English corpus needs a per-document text-search configuration (language detection already exists in the ingestion stage).

### Stage 3: Weighted Reciprocal Rank Fusion (`src/retrieving/fusion.py`)
Cosine similarities and `ts_rank_cd` scores live on unrelated scales and cannot be added. Fusion therefore uses ranks only:

`score(chunk) = dense_weight / (rrf_k + dense_rank) + sparse_weight / (rrf_k + sparse_rank)`, ranks from 1, a missing rank contributing nothing.

A chunk high in both lists rises to the top. The weights and `rrf_k` are search knobs rather than constants, which is why fusion is ~20 lines of our own code instead of a library call. The unweighted case is tested against ranx's RRF implementation. Fused scores are scaled so the best is 1.0; they order results and are not comparable across queries.

RRF fuses on `chunk_id`, so both lists must use the same identifier. Until the Postgres migration they did not: dense results carried the vector store's UUID point ID while sparse results carried the real chunk ID. A chunk found by both retrievers therefore never received a fused score, and it could occupy two of the five result slots. Measured on the benchmark's 38 queries: 27 of 190 top-5 slots (14.2%) held a duplicate, so the generator saw four distinct chunks instead of five in 27 queries. Recall did not change (see Phase 6 for why the benchmark could not detect it).

**Filtered-search settings on Neon.** The HNSW settings above must reach every search connection. Neon's proxy silently drops individual startup parameters (tested with `work_mem`, which was ignored) but forwards libpq's `options` parameter, so the settings are sent as `options=-chnsw.iterative_scan=relaxed_order -chnsw.ef_search=100`. An integration test asserts the values actually arrive in the session. Without it, filtered search would have silently lost iterative scans.

### Stage 4: Optional Cross-Encoder Reranking (`src/retrieving/rerankers/`)
A cross-encoder reads the query and each candidate together and scores their joint relevance, which embeddings (computed separately) cannot. It reorders the first-stage pool (`rerank_candidates`, chat uses `top_k × 4`) down to `top_k`.

| Reranker | Runs | Cost | Notes |
|---|---|---|---|
| **FlashRank** (default) | In the API process, ONNX on CPU, no torch | $0 | Model `ms-marco-TinyBERT-L-2-v2`, baked into the API image. Reads up to 512 tokens per passage |
| **Jina** | Hosted API | Draws on Jina's one-time grant | `jina-reranker-v2-base-multilingual` |
| **Voyage** | Hosted API | 200M free tokens, then $0.05 per 1M (`rerank-3`) | Paced to its own limit, below |

**Voyage's card-free limit, measured** (2026-10-02, on this project's key): rerank is limited to **3 requests and 10K tokens a minute**, a bucket separate from embeddings (an embedding call succeeded while rerank was rate-limited). Voyage counts the documents' tokens plus the query once per document: a pool of 8 passages of about 600 tokens cost 3,896 tokens and was accepted, while a pool of 20 (about 10K tokens) was rejected with a 429. So at that limit the reranker is only usable with a small `rerank_candidates` (about 8) and one query every 20 seconds or so. Requests are paced with the same window class as embeddings; a 429 waits one request slot (60 s / requests per minute) and is retried twice, then raises, so the pipeline records the run as degraded rather than silently returning the first-stage order. A pool whose estimated tokens exceed the per-minute limit is refused before any request, because it can only be rejected again. Both HTTP rerankers use the shared retry loop (`src/retry.py`); the Jina reranker retries only transient statuses and fails at once on a client error such as a bad key. Adding a payment method would lift these limits, which the spend rule excludes.

**Model choice, measured** (2026-09-29, 20 candidates of ~600 tokens, 16 CPU threads): TinyBERT-L-2 ~95 ms, MiniLM-L-12 ~1.95 s per rerank; both ranked the answering passage first in the spike. The free API host has 0.1 vCPU, where MiniLM would take on the order of 20 seconds, so TinyBERT is the default and MiniLM is one setting away (`FLASHRANK_MODEL`) for an evaluation that measures the quality/latency trade-off. FlashRank's hosted weights are CC-BY-SA and trained on MS MARCO, whose terms are non-commercial.

**Reranker scores order results; they are not relevance probabilities.** A live check (2026-09-29) on three real PEP 20 chunks for "errors should never pass silently": both models ranked the chunk containing that sentence first, but TinyBERT scored it 0.0013 where MiniLM scored it 0.93 (the chunk is 406 tokens, so nothing was truncated). TinyBERT's scores are poorly calibrated, so no score floor (`min_similarity_score`) may be applied to reranked results; the default floor is 0 for this reason.

A local `bge-reranker-v2-m3` (for the optional GPU worker) are future options; each is one more class with the same `rerank` method.

### Score Calibration
Relevance scores are treated as **ranking signals, not absolute cutoffs**. The system defaults to maximizing recall by allowing all retrieved top candidates through, rather than applying an arbitrary minimum score cutoff that might accidentally filter out the correct answer.

### Latency Observability
The retrieval pipeline is highly instrumented, reporting discrete timing for each micro-stage (embedding the query, searching the databases, reranking). This allows operators to easily identify performance bottlenecks in production.

## Design Philosophy & Tradeoffs
- **Reranker: Latency AND Accuracy Tradeoff:** (Measured on the prototype corpus, retired 2026-09-28; FlashRank has not yet been measured on a benchmark.) Contrary to the common assumption that adding a cross-encoder reranker always improves results, empirical ablation against our multi-domain benchmark revealed a concrete regression: the **Jina AI Reranker** (`jina-reranker-v2-base-multilingual`) lowered Recall@1 from **0.974 (Hybrid alone) to 0.816**, while maintaining 1.000 Recall@5. The root cause was the off-the-shelf reranker over-indexing on general summary overview chunks rather than specific technical sub-sections. The reranker is therefore exposed as an **optional runtime toggle** — recommended only for non-interactive, latency-tolerant tasks where deeper cross-attention across the top-5 candidates is more valuable than pinpoint top-1 precision.
