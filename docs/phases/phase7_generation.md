# Phase 7: Generation

## Overview
The generation phase is responsible for synthesizing the retrieved context into a coherent, accurate, and perfectly cited answer for the user. The primary design philosophy of this phase is strict hallucination prevention: the system must act exclusively as a synthesizer of the provided context and never rely on the generative model's internal pre-training knowledge.

## Core Implementation Logic

### Context Packaging (`src/generating/context_builder.py`)
Before the model is invoked, the retrieved chunks are assembled into one context window, in retrieval order:
- **Exact duplicates are dropped.** Chunks whose text is identical (the same passage indexed from two pages) are included once, by a hash of the text. Nothing is merged or rewritten.
- **A token budget** (`max_context_tokens`, 5,000 by default, estimated from word counts) stops adding chunks once full. Every dropped chunk is recorded as excluded, so what the model did *not* see is inspectable too.
- **An optional score floor** (`min_similarity_score`, 0 by default) excludes low-scoring chunks. It stays at 0 unless the score scale is known to be calibrated; if it excludes everything, the answer is an explicit diagnostic that points at retrieval, not an LLM call.
- **Each chunk carries a citation header**, `[Source: <url> | Section: <heading path>]`, and chunks are separated by `---`.

### Grounding Prompt (`src/generating/prompt_template.py`)
The prompt has four parts: grounding rules, the last five turns of conversation history (when given), the context between `=== CONTEXT START ===` and `=== CONTEXT END ===` markers, and the question. The rules tell the model to answer only from the context, to cite the `[Source: ...]` markers, and to say "I don't have enough information in the provided context to answer this question." when the context does not contain the answer.

### LLM Calls, Fallback and Per-Call Accounting (`src/generating/llm_client.py`)
Calls go through LiteLLM with our own retry and fallback: transient errors (429, 5xx, connection) are retried with backoff, then the configured fallback model answers; a dead model or a timeout falls back immediately. Streaming falls back only before the first token.

Every call returns its own `LLMCall` record: the text, prompt and completion tokens, cost, and the provider and model that **actually answered** (after a fallback, the fallback's). One client serves every concurrent request, so this record is per call, never stored on the client: when usage lived on the shared client, concurrent queries could log one another's tokens and cost. A streamed answer fills a record owned by its request as tokens arrive; if the provider sends no usage block, tokens are estimated at ~4 characters per token.

### Streaming Generation
Answers stream to the client as Server-Sent Events: `token` events as the model produces them, then `sources`, then `done` (and `faithfulness` when requested). The context window and prompt are built once, before streaming starts, and the same context yields the source list. Usage, cost and the serving provider are written to `query_logs` when the stream completes.

## Design Philosophy & Tradeoffs
- **Strictness vs. Helpfulness:** The prompt's extreme strictness against using outside knowledge means the system might occasionally refuse to answer a question that the underlying LLM actually knows the answer to, simply because it wasn't in the retrieved documents. This is an intentional tradeoff: in a production enterprise environment, failing to answer is vastly preferred over confidently hallucinating incorrect information.
- **Context Window Limits:** The system must carefully balance the number of retrieved chunks sent to the generative model. Sending too few hurts accuracy, but sending too many risks overwhelming the model's attention mechanism (the "lost in the middle" phenomenon) and driving up API costs.
