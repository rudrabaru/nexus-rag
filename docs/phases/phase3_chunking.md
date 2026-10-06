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

### Soft Targets and Hard Limits
While the engine prioritizes semantic coherence, it respects the physical constraints of downstream embedding models.
- **Target Size:** The system aims for an optimal chunk size that balances contextual density with retrieval precision.
- **Maximum Thresholds:** The embedding limit (2,000 tokens) is never exceeded, and nothing is cut off to meet it. A block over the limit (less the overlap that is prepended to the chunk after it) is **split**: text at line boundaries, and inside one over-long line at word boundaries; code at line boundaries, each piece re-fenced; a table at row boundaries, each piece repeating the header row (and the heading line above it), so a row is never read without its column names. Every line reaches the index. A document whose chunking fails is skipped and recorded on the job (`unchunked_sources`, job ends `partial_success` with the reason) instead of vanishing silently.

### Small Chunk Merging
A common issue in structure-based chunking is the creation of fragmented, tiny chunks (e.g., a heading with only a single short sentence beneath it). The system implements an intelligent merging step:
- A small chunk is merged into the one after it only when both share a **heading path**, or the small chunk is the **parent** of the next (a section's introduction and its first subsection). A heading-only chunk always merges into its own body.
- **Siblings are never merged.** Two small sections under the same parent are different topics; joining them gives one chunk two unrelated meanings under a single path, which conflicts with treating the heading hierarchy as a retrieval signal. A small chunk with no related neighbour stays small: a short section is a valid chunk, and the `[document > heading path]` prefix gives it context when embedded.
- *Measured trade-off (2026-10-03, this repository's 11 Markdown documents, as a stand-in until a real corpus is ingested):* before, 87 chunks with 11 under 150 tokens (median 276); after, 112 chunks with 38 under 150 (median 219). More, smaller chunks is the intended price. *Experiment:* whether it helps or hurts retrieval must be measured on the first real corpus before this rule is called settled.

### Prose Overlap
To maintain context between adjacent chunks and avoid cutting off thoughts abruptly, a controlled overlap is introduced at the boundaries. Crucially, this overlap is restricted to prose; atomic blocks (like code or tables) are explicitly excluded from overlap duplication to prevent noise, redundancy, and artificially inflated similarity scores during retrieval.

### Thresholds And Their Status

| Value | Setting | Rationale / status |
|---|---|---|
| Target chunk size | 600 tokens | Sizing audit (below): the median and p90 of the prototype corpus sat at 574 and 632 with no retrieval defect traced to size. Experiment: not yet compared with other sizes. |
| Maximum chunk size | 800 tokens | A code block or table may push a chunk past the target up to this size to stay whole. Experiment. |
| Overlap | 125 tokens, prose only | Roughly one short paragraph of context across a boundary; code and tables are excluded so structured data is never duplicated. Experiment. |
| Minimum chunk size | 150 tokens | Below this a chunk is a merge candidate (subject to the heading rule above); it is not a rule that small chunks are wrong. Experiment. |
| Tiny chunk | under 40 tokens | A heading with at most a line or two; always merged into its own body. |
| Embedding limit | 2,000 tokens | Well under the embedding model's input limit, leaving room for the heading-path prefix. A block over it is split. |
| Wall of text | 5,000 characters, cut at about 3,000 | A paragraph of about 1,000 words with no blank line to cut at. Experiment: set from one documentation corpus. |

Token counts use `cl100k_base`, with special-token strings such as `<|endoftext|>` encoded as ordinary text (a page about language models mentions them).

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

**Open questions (not yet measured):** whether the missing PDF headings affect any benchmark query; whether the 40-tenant pool let near-identical chunks from different tenants occupy the top-k of the then all-tenant evaluation. (Moot since 2026-10-01: evaluations are scoped to one tenant.)

## Follow-up (item 8, 2026-09-26)

- **(1) PDF structure: addressed upstream, chunker unchanged.** Uploaded PDFs and DOCX files are now parsed by Docling (Phase 1), which emits Markdown headings the existing heading-aware chunker already consumes. On the spike documents every PDF produced headings (6 to 44 per document), where the audit found 97.6% of PDF chunks had no heading path. The chunk-level effect on the live corpus has to be re-measured after re-ingestion; PDFs over the Docling page cap still arrive as plain text without headings.
- **(2) Cross-document boilerplate: still open.** Reader APIs now return the main page content (Jina Reader; Firecrawl with `onlyMainContent`), which may remove some navigation, but no cross-document filter was added. It needs the baseline comparison first, and that comparison needs the re-ingested corpus.
- **The chunker was not replaced** (the plan's Docling HybridChunker / Chonkie option), because the audit found nothing wrong with chunk sizing or boundaries.
