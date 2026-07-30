# Phase 7: Generation

## Overview
The generation phase is responsible for synthesizing the retrieved context into a coherent, accurate, and perfectly cited answer for the user. The primary design philosophy of this phase is strict hallucination prevention: the system must act exclusively as a synthesizer of the provided context and never rely on the generative model's internal pre-training knowledge.

## Core Implementation Logic

### Context Packaging & Deduplication
Before the generative model is invoked, the retrieved chunks must be formatted into a clean context window.
- **Deduplication:** The system scans the retrieved chunks and merges adjacent text blocks originating from the same document. This maximizes the utilization of the context window by eliminating redundant overlap.
- **Formatting:** Each distinct block of context is injected into the prompt surrounded by clear XML-style tags, labeled with a specific citation ID (e.g., `[Doc 1]`). This creates a rigid boundary between what is "context" and what is "instruction".

### Strict Anti-Hallucination Prompts
The core system prompt is engineered to enforce absolute adherence to the provided context. The instructions explicitly command the model to:
1. Act as a precise technical assistant.
2. Answer the question using *only* the provided context blocks.
3. Completely ignore its own internal knowledge base.
4. Openly admit "I don't know" if the answer cannot be found in the provided context, rather than attempting to guess or extrapolate.

### Enforced Citation Mechanics
The prompt enforces a strict citation format. Every factual claim made in the generated answer must be immediately followed by the specific citation ID of the context block that supports it (e.g., `... as detailed in the setup guide [Doc 2].`). 

This allows end-users to easily trace any claim back to the exact source document, building trust in the system's outputs.

### Streaming Generation & Token Observability
To ensure a highly responsive user experience, the generation phase supports real-time streaming. As the generative model produces tokens, they are immediately streamed back to the client via Server-Sent Events (SSE). This reduces the perceived latency of the system to near-zero, even for complex, multi-paragraph answers.

Each LLM provider surfaces usage metadata (prompt and completion token counts) from within its own stream chunks. The pipeline accumulates these token counts during streaming and commits them to the observability registry upon completion. This ensures accurate cost tracking even in streaming mode — a non-trivial problem, since streaming generators complete asynchronously and cannot be naively awaited for a final usage summary.

## Design Philosophy & Tradeoffs
- **Strictness vs. Helpfulness:** The prompt's extreme strictness against using outside knowledge means the system might occasionally refuse to answer a question that the underlying LLM actually knows the answer to, simply because it wasn't in the retrieved documents. This is an intentional tradeoff: in a production enterprise environment, failing to answer is vastly preferred over confidently hallucinating incorrect information.
- **Context Window Limits:** The system must carefully balance the number of retrieved chunks sent to the generative model. Sending too few hurts accuracy, but sending too many risks overwhelming the model's attention mechanism (the "lost in the middle" phenomenon) and driving up API costs.
