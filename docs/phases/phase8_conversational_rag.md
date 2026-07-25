# Phase 8: Conversational RAG

## Overview
A standalone RAG pipeline is stateless; it treats every query in isolation. However, real-world users interact conversationally, often using pronouns or omitting context in follow-up questions (e.g., "How do I install it?" or "What about the other option?"). The Conversational RAG phase introduces memory and state, allowing the system to handle complex, multi-turn dialogues.

## Core Implementation Logic

### Stateful Query Rewriting
When a user submits a query within an active chat session, the system intercepts the query before it reaches the retrieval phase.

1. **Context Analysis:** The system examines the user's raw query alongside the history of the current conversation (the preceding questions and answers).
2. **Intent Detection:** A specialized, high-speed LLM evaluates whether the new query is a standalone question or a follow-up that relies on previous context.
3. **Query Formulation:** If it is a follow-up, the rewriter model synthesizes a new, fully self-contained search query. For example, it translates "How do I install it?" into "How do I install the PostgreSQL database server?" based on the prior chat turns.

This rewritten query is then passed to the retrieval engine (Phase 5), ensuring the search algorithms receive explicit, unambiguous keywords and semantic concepts.

### Memory Management
To prevent the context window from growing infinitely and slowing down the rewriter model, the system manages conversation history dynamically.
- Only the most recent, relevant turns of the conversation are sent to the query rewriter.
- This sliding window approach guarantees high performance while maintaining enough context to resolve immediate conversational references.

### Transparent Processing
The query rewriting process happens entirely behind the scenes. However, for observability and debugging, the system logs both the raw user query and the expanded rewritten query. The generation phase (Phase 7) is then fed the retrieved documents alongside the *rewritten* query, ensuring the final answer perfectly aligns with the user's implicit intent.

## Design Philosophy & Tradeoffs
- **Latency vs. Accuracy:** Query rewriting requires an additional LLM call before retrieval can even begin, inherently adding latency to the overall pipeline. To minimize this, the system routes rewriting tasks to exceptionally fast, lightweight models optimized for speed rather than deep reasoning.
- **Aggressive Rewriting:** If the rewriter model is too aggressive, it might alter the user's intent. To mitigate this, the prompt explicitly instructs the rewriter to act purely as a translator of context, forbidding it from trying to answer the question itself or introducing new concepts not present in the chat history.
