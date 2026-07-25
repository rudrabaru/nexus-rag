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
