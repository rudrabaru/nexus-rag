# Phase 4: Embedding

## Overview
The embedding phase turns each chunk into a vector so that semantically similar text can be found for a query. Embedding is **API-first and provider-swappable**: the model is chosen per *index*, never mixed within one, and no embedding model runs inside the API process.

## Core Implementation Logic

### One index, one embedding model
Vectors from different models live in different spaces; a cosine similarity between a Jina vector and a Voyage vector is meaningless. The system therefore makes the model a property of an **index**, identified as `provider:model` (for example `voyage:voyage-4`):

- Every chunk row records its `index_id` and `embedding_model`, and every index has a row in `embedding_indexes` (provider, model, dimension).
- Every search is scoped to one index, dense **and** sparse. Keeping the keyword search on the same rows means hybrid fusion never mixes two copies of the same chunk.
- The chunk key is `(tenant_id, index_id, chunk_id)`, so the same corpus can exist in two indexes at once. That is what makes an embedding comparison a controlled experiment: same chunks, same text, only the model differs.
- A query embedder must match the store's index; constructing a retriever with a mismatched pair fails immediately rather than returning nonsense.
- Embedding is **not** a search dimension in evaluation sweeps. Changing it means building a new index, which is done once per corpus, not per query.

### Providers
Providers differ only in wire format, so each is a small function that builds an `Embedder` (endpoint, request body, response reader, limits) rather than a class in a hierarchy (`src/embedding/providers.py`).

| Provider | Default model | Role | Free allowance |
|---|---|---|---|
| **Voyage** (default) | `voyage-4` (1024-dim) | Hosted default | 200M tokens per model, one-time. **3 requests/min and 10K tokens/min without a payment method** (verified 2026-09-26); adding a payment method raises the limits and the free tokens still apply |
| **Ollama** | `bge-m3` (1024-dim) | Local index for the optional GPU worker | Unlimited, local |
| **Jina** (legacy) | `jina-embeddings-v3` | Only to keep the prototype index the frozen baselines were measured on queryable | A one-time grant shared with the reranker |

All three produce 1024-dimensional vectors, the width of the `chunks.embedding` column. A model with another width is refused with an explicit error instead of failing on insert.

### Asymmetric encoding
Retrieval models embed queries and documents differently (Voyage `input_type`, Jina `task`). Callers always state which side they are embedding: chunks as `document`, queries as `query`. Using the wrong side silently lowers recall, so it is an explicit argument, not a default.

### What is embedded
The embedded text is the chunk prefixed with its document and heading path: `[Document > Section > Subsection]\n<chunk text>`. The prefix gives short chunks the context of where they sit.

### Pacing, retries and partial success
- **Client-side pacing.** Voyage's no-payment-method limits (3 RPM / 10K TPM) are enforced before sending, with a sliding one-minute window. Sending as fast as possible and backing off on 429s would spend the same wall-clock time in penalties. The window is thread-based, not asyncio-based, because each ingestion job runs its own event loop. Token counts for pacing are estimated at ~3 characters per token (over-estimating English) because the API image carries no tokenizer.
- **Request splitting.** No request exceeds the per-request token cap or the per-minute window: a request larger than the window could never be admitted.
- **Retries.** 408/429/5xx and network errors are retried with backoff, honouring `Retry-After`. Errors retrying cannot fix (a bad key, an unknown model) fail immediately.
- **Partial success.** A batch that still fails is recorded by chunk index; the document commits the chunks that did embed and the job ends as `partial_success` with the reason. A run where nothing embedded raises, so the job is retried.

### Switching the embedding model
Changing `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` points both ingestion and search at a different index, which starts empty: documents are re-ingested with the new model rather than copied between indexes. The API warns at startup when its index holds no chunks. (A copy-between-indexes tool existed for moving the prototype corpus from Jina to Voyage; it was removed when that corpus was retired as prototype data.)

### Storage: one row per chunk in Postgres
Each embedded chunk is one row of `chunks` in Postgres (Neon): text, vector (`halfvec(1024)`, HNSW) and a generated full-text column (`tsvector`, GIN).
- **One write, two indexes.** The keyword index is computed from the same row, so dense and sparse cannot diverge.
- **Idempotent upserts** on `(tenant_id, index_id, chunk_id)`.
- **Cascading deletes** from the document.
- **`halfvec`** halves storage (~2 KB per chunk) under Neon's 0.5 GB free tier; round-trip cosine stays above 0.9999.
- **One HNSW graph for all indexes.** Searches filter by `index_id` with iterative scans. With a second large index, a partial HNSW index per `index_id` would keep each graph model-pure; at one active index plus a transitional copy this is not yet worth the DDL.

## Design Philosophy & Tradeoffs
- **Why Voyage over Jina:** Jina's free allowance is a one-time grant shared by embeddings, reranking and the reader. It drains permanently. Voyage's grant is larger (200M tokens per model; a 2,000-chunk corpus is about 1.4M tokens).
- **The cost of card-free Voyage:** 3 RPM is shared by the whole account, so query embedding (one request per uncached query) competes with ingestion, and a chat burst beyond 3 queries/minute waits for the window. Adding a payment method removes this without spending money while the free tokens last. The local Ollama index is the alternative that has no limit at all.
- **Network dependency:** hosted embedding needs outbound calls; there is no silent local fallback, because falling back to a different model would put vectors from two spaces into one index.
- **Known gap:** the query-embedding cache reports zero tokens on a hit, so a later configuration in a sweep looks cheaper than an earlier one. Addressed with the evaluation engine (item 10).
