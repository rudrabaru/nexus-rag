# Phase 3: Chunking

## Overview
Chunking is the process of breaking down cleaned, normalized documents into smaller, semantically coherent segments suitable for vector embedding and retrieval. The primary design philosophy is that chunk boundaries should be dictated by the logical structure of the document (headings, paragraphs, code blocks) rather than arbitrary token counts, ensuring high-fidelity semantic retention.

## Core Implementation Logic

### Semantic Hierarchy Preservation
The chunking engine treats Markdown as a tree structure rather than a flat string. 
- **Section-Based Splitting:** Documents are split at heading boundaries. This guarantees that chunks align closely with the author's original topics.
- **Path Tracking:** As the document is chunked, the system maintains a "heading path" (e.g., `["Introduction", "Setup", "Installation"]`). This breadcrumb path is injected into the chunk's metadata, providing downstream retrievers and the generation model with crucial context about where the chunk originated in the broader document hierarchy.

### Block Atomicity
Certain structural elements must remain completely intact to preserve their semantic meaning. The system enforces strict atomic boundaries for:
- **Code Blocks:** A fenced block of code is never split mid-block, preventing syntax corruption or broken logic.
- **Tables:** Markdown tables are kept contiguous, ensuring rows and columns are not separated across different chunks, which would destroy their relational meaning.
- **Visuals:** Any described images or visual references are grouped as atomic, distinct multimodal units.

### Soft Targets and Hard Limits
While the engine prioritizes semantic coherence, it respects the physical constraints of downstream embedding models.
- **Target Size:** The system aims for an optimal chunk size that balances contextual density with retrieval precision.
- **Maximum Thresholds:** A strict upper token limit is enforced. If an atomic section naturally exceeds this limit, it falls back to a secondary splitting strategy, carefully breaking long passages by natural paragraph boundaries or single newlines as a last resort.

### Small Chunk Merging
A common issue in structure-based chunking is the creation of fragmented, tiny chunks (e.g., a heading with only a single short sentence beneath it). The system implements an intelligent merging step:
- It aggregates small, adjacent chunks that share the same parent heading hierarchy until they reach the optimal target size.
- This prevents sparse chunks that lack sufficient context for accurate similarity matching.

### Prose Overlap
To maintain context between adjacent chunks and avoid cutting off thoughts abruptly, a controlled overlap is introduced at the boundaries. Crucially, this overlap is restricted to prose; atomic blocks (like code or tables) are explicitly excluded from overlap duplication to prevent noise, redundancy, and artificially inflated similarity scores during retrieval.

## Corpus Audit (2026-09-22)

**Why:** before replacing any parsing or chunking stage, quantify what is actually wrong. The audit read every chunk in the live collection (read-only) and computed structural statistics only; no keyword or site-specific rule was used.

**Corpus:** 2,284 chunks, 62 documents, 40 tenant workspaces. Sources: 1,255 web chunks, 929 uploads (497 PDF, 367 TXT, 49 MD, 16 DOCX), 100 with no recognisable source URL.

| Finding | Measurement | Verdict |
|---|---|---|
| Chunk size | median 574 tokens, p90 632, p99 907; 1.5% under 150, 1.6% over 800, 7 chunks over 1,200 (max 3,113) | Sizing works; no change justified. |
| Heading hierarchy | 53.8% of chunks have no heading path: 100% of TXT (plain text has no headings), **97.6% of PDF**, 29.4% of web, 6-10% of DOCX/MD | **PDF structure is not being recovered.** This is measured evidence for improving PDF parsing, and it is the only parsing change the audit supports. |
| Repeated text | 39.0% of chunks are exact repeats of another chunk, but only **14.6%** repeat within the same tenant; the rest are legitimate copies across tenants | Cross-tenant copies are expected and not a defect. |
| Boilerplate | Of 100 same-tenant repeat groups, **71 span more than one document** (cookie-consent blocks, image-link lists, repeated navigation), up to 11 documents per group; the shortest chunks are repeated 3-token headings | **Web ingestion indexes site boilerplate.** The existing boilerplate filter compares blocks only within one ingestion job, so repeats across separate documents survive. |

**Conclusions.** The current chunker is not the problem. Two upstream causes are supported by evidence: (1) PDF text extraction loses heading structure; (2) boilerplate that repeats across documents is indexed. Neither justifies replacing the chunker. A generic, structural filter (block text repeated across many documents of one tenant) is the indicated fix for (2); it must be validated against the baselines in the evaluation phase before adoption.

**Open questions (not yet measured):** whether the missing PDF headings affect any benchmark query; whether the 40-tenant pool lets near-identical chunks from different tenants occupy the top-k of an all-tenant evaluation (the evaluator searches with `allow_global=True`).
