# Phase 7: Generation

## Overview
The generation phase is responsible for synthesizing the retrieved context into a coherent, accurate, and perfectly cited answer for the user. The primary design philosophy of this phase is strict hallucination prevention: the system must act exclusively as a synthesizer of the provided context and never rely on the generative model's internal pre-training knowledge.

## Core Implementation Logic

### Context Packaging (`src/generating/context_builder.py`)
Before the model is invoked, the retrieved chunks are assembled into one context window, in retrieval order:
- **Exact duplicates are dropped.** Chunks whose text is identical (the same passage indexed from two pages) are included once, by a hash of the text. Nothing is merged or rewritten.
- **A token budget** (`max_context_tokens`, 5,000 by default) stops adding chunks once full. A chunk costs its real tiktoken count from the chunker (a row without one falls back to the shared estimator, 3 characters per token). 5,000 keeps one request near 5.5K tokens, which fits Groq's free 8K tokens a minute; it is an experiment knob, not tuned on retrieval results. Every chunk that is left out, whether below the score floor, a repeat of an included chunk, or over the budget, is recorded as excluded with its reason, so what the model did *not* see is inspectable too.
- **An optional score floor** (`min_similarity_score`, 0 by default) excludes low-scoring chunks. It stays at 0 unless the score scale is known to be calibrated; if nothing is left (including when nothing was retrieved), no model call is made and the answer is a diagnostic that names the actual reason: nothing retrieved, below the floor, over the budget or repeated.
- **Each chunk carries a citation header**, `[Source: <url> | Section: <heading path>]`, and chunks are separated by `---`.

### Grounding Prompt (`src/generating/prompt_template.py`)
The prompt has four parts: grounding rules, the last five turns of conversation history (when given), the context between `=== CONTEXT START ===` and `=== CONTEXT END ===` markers, and the question. The rules tell the model to answer only from the context and to say "I don't have enough information in the provided context to answer this question." when the context does not contain the answer. The instruction to cite the `[Source: ...]` markers is added only when `cite_sources` is on.

### LLM Calls, Fallback and Per-Call Accounting (`src/llm/`)
Calls go through LiteLLM with our own retry and fallback, and one classifier (`src/llm/errors.py`) serves both the streaming and the non-streaming path: transient errors (429, 5xx, connection, an empty body or an empty stream) are retried with backoff, then the configured fallback model answers; a dead model, a bad key or a timeout falls back immediately. Streaming falls back only before the first token. When retries and the fallback are spent the call raises `GenerationError`; a failure is never returned as answer text.

Providers are `gemini`, `groq` and `openai`; each needs only its own API key. Models are assigned to roles in settings, as `provider/model` (`src/llm/roles.py`): `LLM_CHAT` (with an optional `LLM_CHAT_FALLBACK`, used when that provider's key is set), `LLM_REWRITE` (falls back to the chat model), `LLM_JUDGE` and `LLM_TESTSET` (both pinned: no fallback, because a different model mid-run changes what is measured). `src/config_checks.py` fails fast on a malformed role or a missing chat key. Gemini carries chat and test-set generation; Groq's `gpt-oss` models judge and handle short prompts, because its free tier (8K tokens a minute, 200K a day) is too small for long prompts, and the judge differs in family from the generator. An experiment pins its models and never falls back. `extract_json_object` (`src/llm/structured.py`) reads a JSON object out of a reply that may be wrapped in prose or a code fence; the faithfulness judge and the test-set generator share it.

Every call returns its own `LLMCall` record: the text, prompt and completion tokens, cost, and the provider and model that **actually answered** (after a fallback, the fallback's). One client serves every concurrent request, so this record is per call, never stored on the client: when usage lived on the shared client, concurrent queries could log one another's tokens and cost. A streamed answer fills a record owned by its request as tokens arrive; if the provider sends no usage block, tokens are estimated by the shared estimator (`src/tokens.py`, 3 characters per token, which over-estimates English so budgets and pacing err safe). Cost comes from LiteLLM's price data; when it has no price for a model the call is marked `cost_known: false`, so an unpriced model is never mistaken for a free one.

### Streaming Generation
Answers stream to the client as Server-Sent Events: `token` events as the model produces them, then `sources`, then `done` (and `faithfulness` when requested). The context window and prompt are built once, before streaming starts, and the same context yields the source list. Usage, cost and the serving provider are written to `query_logs` when the stream completes.

### The Faithfulness Judge
The judge scores 0, 0.5 or 1 and nothing else: a reply with another number, NaN or no number is rejected as unusable instead of being averaged in. Evaluations use the strict form, where an unavailable judge or an unusable reply raises and is never recorded as a score. Chat uses the lenient form, where a failure leaves the score empty (null) with the reason; it is never recorded as 0.0, which would read as "the answer was a hallucination".

### Query Rewriting
A follow-up is made standalone using the last six messages of history, and the query can optionally be generalised (`ENABLE_QUERY_GENERALISATION`, off by default). Generalising changes what is retrieved but is not part of what an experiment measures, so chat does not use it unless chosen; its prompt names no corpus or domain. An empty or unreadable rewrite keeps the original query.

## Design Philosophy & Tradeoffs
- **Strictness vs. Helpfulness:** The prompt's extreme strictness against using outside knowledge means the system might occasionally refuse to answer a question that the underlying LLM actually knows the answer to, simply because it wasn't in the retrieved documents. This is an intentional tradeoff: in a production enterprise environment, failing to answer is vastly preferred over confidently hallucinating incorrect information.
- **Context Window Limits:** The system must carefully balance the number of retrieved chunks sent to the generative model. Sending too few hurts accuracy, but sending too many risks overwhelming the model's attention mechanism (the "lost in the middle" phenomenon) and driving up API costs.
