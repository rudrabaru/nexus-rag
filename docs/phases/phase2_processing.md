# Phase 2: Processing, Cleaning & Normalization

## Overview
Once raw documents are converted to Markdown by the ingestion phase, they enter the Processing phase. The objective is to remove statistical noise and boilerplate before chunking, ensuring only dense, relevant information is indexed. All removal decisions are driven by measurable structural evidence — never by hardcoded keyword matching.

## Core Implementation Logic

### Structural Block Parsing
The cleaning engine first parses each document's Markdown into a list of atomic blocks. Blocks are delineated by logical boundaries:
- Code fences
- Markdown tables
- Heading lines
- Paragraph breaks

Code blocks and tables are isolated as placeholder tokens first, then re-injected after paragraph splitting. This prevents multi-line structures (like scripts or data tables) from being accidentally split during processing.

### Content Hashing & Frequency Analysis
Each block is normalized and cryptographically hashed for deduplication:
- Dynamic elements like URLs inside Markdown links are stripped before hashing.
- Dates are normalized to avoid churn from time-stamped, rotating blocks.
- Whitespace is collapsed.

The system tracks how often each unique block appears across the ingested corpus. Blocks that appear repeatedly across a large proportion of documents are flagged as likely boilerplate.

**PDF page furniture.** Running page headers and footers of uploaded PDFs are removed before this stage: Docling's layout model labels them as page furniture, and they are excluded from its Markdown export. The plain-text fallback (PDFs over the Docling page cap, or that Docling fails on) keeps them. An earlier filter meant to catch them here (short blocks repeated 3+ times within one PDF) never took effect: it flagged blocks that the cleaner then re-scored and kept, and it only inflated the "blocks removed" audit count. It was deleted (2026-09-28) rather than switched on, because switching it on would remove content with no measurement behind it; that needs a before/after comparison on fallback PDFs first.

### Block Metrics & Removal Thresholds
Every block is evaluated using measurable signals:
- **Link Density**: The ratio of link characters to total text characters.
- **Word Count**: The absolute length of the block.
- **Document Frequency**: The fraction of the total corpus where this block appears.

A block is removed if it passes an objective, data-driven threshold:
- High link density combined with low word count strongly indicates a navigation menu or footer.
- High document frequency combined with low word count strongly indicates repetitive boilerplate (e.g., copyright notices or site-wide banners).

> **Corpus-Independence Rule:** No specific text, heading title, or keyword (e.g., "Related Links") is ever hardcoded as a removal trigger. Removal is always driven by statistical evidence from the corpus itself.

### Inspecting What Was Removed
Every document logs its word count before and after cleaning and how many blocks were removed (`CLEAN | <url> | words 1200 -> 1105 | blocks removed 4/61`). At DEBUG level each removed block is logged with the signals that removed it and its first 80 characters, so a removal can always be traced to measured evidence. When cleaning leaves under 50 characters, the raw Markdown is chunked instead: that outcome is treated as a cleaner failure, not as an empty document.

### Content Preservation Philosophy
The overarching principle is: **when in doubt, preserve**. A small amount of noise retained is far less costly than accidentally removing genuine content. The thresholds above are intentionally conservative and biased toward false negatives (keeping unimportant text) rather than false positives (removing important text).

## Design Philosophy & Tradeoffs
- **Batch-Level vs. Stream-Level Analysis:** Corpus-frequency analysis requires seeing a sufficient batch of documents to identify what is truly "boilerplate" versus unique content. For single-document ingestions, the system gracefully falls back to relying primarily on structural signals (like link density) rather than cross-document frequency.
- **Single-Pass Cleaning:** The cleaner operates as a single-pass algorithm to maintain the high throughput required of a real-time microservice, opting against multi-pass iterative cleaning that would slow down ingestion.
