---
title: Nexus RAG
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# NexusRAG

An advanced, highly performant Retrieval-Augmented Generation (RAG) system engineered as an in-memory microservice. Designed for extreme accuracy, strict hallucination prevention, and absolute corpus independence.

## Architecture Vision
The system ingests heterogeneous data formats and transforms them into an intelligent, vector-searchable knowledge base. It adheres strictly to in-memory processing patterns—avoiding slow intermediate disk writes—while maintaining comprehensive pipeline observability and strict multi-tenant security isolation. Built for the cloud, it gracefully handles complex, stateful human conversations at scale.

## Enterprise Features

### Intelligent Data Ingestion
- **Multi-Format Extraction:** Seamlessly processes URLs, PDFs, DOCX, MD, CSV, and TXT files.
- **Multimodal Vision:** Optional OCR and visual analysis intercepts images and scanned documents, converting visual data into searchable text.
- **Dynamic Web & Sitemap Crawling:** Employs headless browser rendering and intelligent Jina Reader integrations. Supports selective XML sitemap URL prefix filtering for targeted subsection indexing.
- **Bounded Concurrency & Observability:** Features semaphore-based asynchronous concurrency control (`asyncio.Semaphore`) and surfaces granular root-cause error reasons (e.g., HTTP 429 rate limits, bot blocks) in UI and metadata for partial ingestion failures.

### Semantic Normalization & Chunking
- **Boilerplate Detection:** Uses statistical corpus frequency analysis to automatically detect and strip navigation menus, footers, and noise. For single-document ingestions, gracefully falls back to structural signals (link density, word count, edge position) when corpus statistics aren't available.
- **Hierarchical Chunking:** Preserves semantic meaning by chunking documents based on their underlying structural hierarchy (headings, paragraphs) rather than arbitrary token counts. 
- **Block Atomicity:** Enforces strict contiguous boundaries for code blocks, tables, and images, guaranteeing that complex structures are never split during processing.

### Advanced Retrieval Engine
- **Hybrid Search:** Combines semantic dense vector search with a dynamic sparse keyword index to capture both conceptual intent and exact terminology.
- **Reciprocal Rank Fusion (RRF):** Fuses the results of both search paradigms mathematically for optimal recall.
- **Cross-Encoder Reranking:** Applies a high-fidelity cross-encoder model to surface the most contextually relevant chunks from the fused candidate pool.
- **Multi-Tenancy & Isolation:** Deep integration of Tenant ID and Visibility scopes at the database layer ensures private workspace data is cryptographically isolated and invisible to unauthorized queries.

### Conversational Memory & Generation
- **Stateful Query Rewriting:** Intercepts human follow-up questions and automatically rewrites them into fully resolved search queries based on the conversation history.
- **Strict Anti-Hallucination Constraints:** Employs aggressive prompt engineering to force the generative model to answer *only* from the provided context.
- **Explicit Citation Tracking:** Every factual claim generated is explicitly cited and mapped back to the specific source document and section heading.
- **Real-Time Streaming:** Streams tokens back to the client via Server-Sent Events (SSE) for a near-zero latency UX.

### Observability & Automated Evaluation
- **Pipeline Telemetry:** Built-in observability logs discrete timing for every micro-stage (embedding, search, reranking, generation) to pinpoint latency bottlenecks.
- **LLM-as-a-Judge Evaluation:** Features an automated evaluation framework that synthesizes evaluation datasets directly from your corpus, grading the pipeline on retrieval recall and generation faithfulness.

## Pipeline Architecture

Detailed architectural and implementation documentation for each phase of the pipeline can be found in the `docs/phases/` directory. These documents explain the core logic and design philosophy behind the system.

1. [Phase 1: Ingestion](docs/phases/phase1_ingestion.md) - Multi-format data extraction and normalization.
2. [Phase 2: Processing](docs/phases/phase2_processing.md) - Boilerplate cleaning, noise removal, and structural validation.
3. [Phase 3: Chunking](docs/phases/phase3_chunking.md) - Hierarchical semantic splitting and block atomicity.
4. [Phase 4: Embedding](docs/phases/phase4_embedding.md) - Dense and sparse vector generation.
5. [Phase 5: Retrieval](docs/phases/phase5_retrieval.md) - RRF Fusion and Cross-Encoder reranking.
6. [Phase 6: Evaluation](docs/phases/phase6_evaluation.md) - Automated LLM-as-a-judge benchmarking.
7. [Phase 7: Generation](docs/phases/phase7_generation.md) - Strict anti-hallucination prompt engineering and citation.
8. [Phase 8: Conversational RAG](docs/phases/phase8_conversational_rag.md) - Stateful query rewriting and memory management.
9. [Phase 9: Production Tradeoffs](docs/phases/phase9_production_tradeoffs.md) - Architectural design decisions and tradeoffs.

## Deployment & Production Notes

NexusRAG is designed to be deployed as a containerized microservice. It is optimized to run efficiently even on resource-constrained cloud infrastructure by offloading heavy ML computation (embeddings and generation) to specialized external APIs, maintaining a near-zero memory footprint for AI models.

**Storage Requirements:**
The system is largely stateless with the exception of the local SQLite registry. For production deployments (e.g., GCP Cloud Run, Render, HuggingFace Spaces), Qdrant Cloud handles persistent vector storage, meaning you can safely restart your backend containers without losing your embedding data. Ensure `QDRANT_URL` and `QDRANT_API_KEY` are securely configured.
