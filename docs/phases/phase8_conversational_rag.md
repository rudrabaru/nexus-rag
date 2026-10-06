# Phase 8: Conversational RAG

## Overview
A standalone RAG pipeline is stateless; it treats every query in isolation. However, real-world users interact conversationally, often using pronouns or omitting context in follow-up questions (e.g., "How do I install it?" or "What about the other option?"). The Conversational RAG phase introduces memory and state, allowing the system to handle complex, multi-turn dialogues.

## Core Implementation Logic

### Stateful Query Rewriting
When a user submits a query within an active chat session, the system intercepts the query before it reaches the retrieval phase.

1. **Context Analysis:** The system examines the user's raw query alongside the history of the current conversation (the preceding questions and answers).
2. **History gate:** With no history the query is searched as written, with no model call. With history, a high-speed LLM is always asked to rewrite it; a question that already stands alone should come back essentially unchanged. There is no separate classifier step.
3. **Query Formulation:** The rewriter model synthesizes a fully self-contained search query. If its reply is empty or unreadable, or the call fails, the original query is searched instead. For example, it translates "How do I install it?" into "How do I install the PostgreSQL database server?" based on the prior chat turns.

This rewritten query is then passed to the retrieval engine (Phase 5), ensuring the search algorithms receive explicit, unambiguous keywords and semantic concepts.

### Memory Management
To prevent the context window from growing infinitely and slowing down the rewriter model, the system manages conversation history dynamically.
- The query rewriter receives the six most recent messages, and the generation prompt carries the five most recent. Both are sliding windows over the history the client sends (the API accepts at most 20 turns of 4,000 characters each), not a relevance selection.
- A window keeps the rewriter fast and cheap while leaving enough context to resolve immediate references; a reference to something older than the window will not be resolved.

### Transparent Processing
The query rewriting process happens entirely behind the scenes. For observability the pipeline events record the original query when a request starts and the rewritten query that was actually searched. The rewritten query is used **only for retrieval**. The generation phase (Phase 7) receives the retrieved chunks, the recent history and the user's **original** question, so the answer addresses what the user asked, not the rewriter's paraphrase of it.

## Design Philosophy & Tradeoffs
- **Latency vs. Accuracy:** Query rewriting requires an additional LLM call before retrieval can even begin, inherently adding latency to the overall pipeline. To minimize this, the system routes rewriting tasks to exceptionally fast, lightweight models optimized for speed rather than deep reasoning.
- **Aggressive Rewriting:** If the rewriter model is too aggressive, it might alter the user's intent. The rewrite prompt tells the model not to answer the query, to return only a structured reply, and to put the entities from the history into a standalone query. Nothing in the pipeline checks that the rewrite preserved the intent, so an over-eager rewrite is a retrieval failure to look for in the logged queries. An empty or unreadable reply falls back to the original query.
