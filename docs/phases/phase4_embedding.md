# Phase 4: Embedding

## Overview
The embedding phase turns each chunk into a vector so that semantically similar text can be found for a query. Embedding is **API-first and provider-swappable**: the model is chosen per *index*, never mixed within one, and no embedding model runs inside the API process.

## Core Implementation Logic

### One index, one embedding model
Vectors from different models live in different spaces; a cosine similarity between a Voyage vector and an Ollama vector is meaningless. The system therefore makes the model a property of an **index**, identified as `provider:model` (for example `voyage:voyage-4`):

- Every chunk row records its `index_id` and `embedding_model`, and every index has a row in `embedding_indexes` (provider, model, dimension).
- Every search is scoped to one index, dense **and** sparse. Keeping the keyword search on the same rows means hybrid fusion never mixes two copies of the same chunk.
- The chunk id starts with the document's id, and the chunk key is `(tenant_id, index_id, chunk_id)`, so the same corpus can exist in two indexes at once. That is what makes an embedding comparison a controlled experiment: same chunks, same text, only the model differs.
- A query embedder must match the store's index; constructing a retriever with a mismatched pair fails immediately rather than returning nonsense.
- Embedding is **not** a search dimension in evaluation sweeps. Changing it means building a new index, which is done once per corpus, not per query.

### Providers
Providers differ only in wire format, so each is a small function that builds an `Embedder` (endpoint, request body, response reader, limits) rather than a class in a hierarchy (`src/embedding/providers.py`).

| Provider | Default model | Role | Free allowance |
|---|---|---|---|
| **Voyage** (default) | `voyage-4` (1024-dim) | Hosted default | 200M tokens per model, one-time. **3 requests/min and 10K tokens/min without a payment method** (verified 2026-09-26); adding a payment method raises the limits and the free tokens still apply |
| **Ollama** | `bge-m3` (1024-dim) | Local index for the optional GPU worker | Unlimited, local |
| **Cloudflare** | `bge-m3` (1024-dim, `@cf/baai/bge-m3`) | Hosted alternative to Voyage, through Workers AI's OpenAI-compatible `/ai/v1/embeddings` | Free plan, no payment method: 10,000 neurons a day at 1,075 neurons per million input tokens, about 9M tokens a day, and 3,000 requests a minute (Cloudflare's pricing and limits pages, read 2026-10-06). Past the daily allowance requests fail until 00:00 UTC |

Both produce 1024-dimensional vectors, the width of the `chunks.embedding` column. A model with another width is refused with an explicit error instead of failing on insert.

**Cloudflare is wired but not yet measured.** The provider (`EMBEDDING_PROVIDER=cloudflare`, with `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`) is tested against a fake server only. Still to establish on a real account: that the output is 1024-dimensional (a wrong width is refused with an error), the real per-request batch and token limits (its documentation states none; the 50 texts and 100,000 tokens per request are conservative experiment values), throughput, and whether `bge-m3` retrieves as well as `voyage-4` on the same chunks. Because each model is its own index, that last question is answered by embedding one corpus twice and comparing the two indexes with the evaluation engine (Phase 6). Queries must use the index's own model, so a Cloudflare index can serve hosted chat queries (unlike a laptop-only Ollama index).

### Asymmetric encoding
Retrieval models embed queries and documents differently (Voyage `input_type`). Callers always state which side they are embedding: chunks as `document`, queries as `query`. Using the wrong side silently lowers recall, so it is an explicit argument, not a default.

### What is embedded
The embedded text is the chunk prefixed with its document and heading path: `[Document > Section > Subsection]\n<chunk text>`. The prefix gives short chunks the context of where they sit.

### Pacing, retries and partial success
- **Client-side pacing.** Voyage's no-payment-method limits (3 RPM / 10K TPM) are enforced before sending, with a sliding one-minute window. Sending as fast as possible and backing off on 429s would spend the same wall-clock time in penalties. The window is thread-based, not asyncio-based, because each ingestion job runs its own event loop. Token counts for pacing are estimated at ~3 characters per token (over-estimating English) because the API image carries no tokenizer.
- **Request splitting.** No request exceeds the per-request token cap or the per-minute window: a request larger than the window could never be admitted.
- **Retries.** One shared loop (`src/retry.py`, also used by the HTTP rerankers) retries 408/5xx and network errors with backoff of 1, 2, 4 seconds, honouring `Retry-After`. A 429 means the minute's budget is spent, so it waits a whole window (or as long as the provider asks): the earlier 2/4/8 s backoff spent every attempt inside one minute. Errors retrying cannot fix (a bad key, an unknown model, a wrong vector width) stop the run at once, with a message that carries the HTTP status but not the provider's text.
- **Checkpoints.** At the card-free Voyage limit one ingestion is hours of paced requests. Every completed request is saved to `embedding_checkpoints` (vector, provider-reported tokens) keyed by job and chunk, with a hash of exactly what was embedded and into which index. A retry of the same job reuses a vector only when that hash matches, so a changed chunker or provider can never reuse a stale one, and sends only the unfinished requests. Checkpoints are deleted in the commit that stores the chunks, or when the job fails. Cost: one staging table; saving a checkpoint is best-effort and never fails the run.
- **Partial success.** A batch that still fails is recorded by chunk index; the document commits the chunks that did embed and the job ends as `partial_success` with the reason. A run where nothing embedded raises, so the job is retried.

### Switching the embedding model
Changing `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` points both ingestion and search at a different index, which starts empty: documents are re-ingested with the new model rather than copied between indexes. The API warns at startup when its index holds no chunks.

### Storage: one row per chunk in Postgres
Each embedded chunk is one row of `chunks` in Postgres (Neon): text, vector (`halfvec(1024)`, HNSW) and a generated full-text column (`tsvector`, GIN).
- **One write, two indexes.** The keyword index is computed from the same row, so dense and sparse cannot diverge.
- **Idempotent upserts** on `(tenant_id, index_id, chunk_id)`.
- **Cascading deletes** from the document.
- **`halfvec`** halves storage (~2 KB per chunk) under Neon's 1 GB free tier; round-trip cosine stays above 0.9999.
- **One HNSW graph for all indexes.** Searches filter by `index_id` with iterative scans. With a second large index, a partial HNSW index per `index_id` would keep each graph model-pure; at one active index plus a transitional copy this is not yet worth the DDL.

## Design Philosophy & Tradeoffs
- **Why Voyage as the default:** the alternative hosted embedder considered (Jina) draws on a one-time grant shared by embeddings, reranking and the reader, which drains permanently. Voyage's grant is larger (200M tokens per model; a 2,000-chunk corpus is about 1.4M tokens).
- **The cost of card-free Voyage:** 3 RPM is shared by the whole account, so query embedding (one request per uncached query) competes with ingestion, and a chat burst beyond 3 queries/minute waits for the window. Adding a payment method removes this without spending money while the free tokens last. The local Ollama index is the alternative that has no limit at all.
- **Network dependency:** hosted embedding needs outbound calls; there is no silent local fallback, because falling back to a different model would put vectors from two spaces into one index.
- **Cost accounting with the query cache:** a cached query embedding costs no tokens, so chat reports zero embedding tokens on a hit. The cache also remembers each query's token count, and the evaluation engine charges every configuration that count, so a later configuration in a sweep is not made to look cheaper than an earlier one (Phase 6).
