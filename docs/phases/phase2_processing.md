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

The system tracks how often each unique block appears across the documents of the **same ingestion job** (one page, one upload, or the pages of one sitemap), not across everything ever ingested. Blocks that appear repeatedly across a large proportion of those documents are flagged as likely boilerplate.

**PDF page furniture.** Running page headers and footers of uploaded PDFs are removed before this stage: Docling's layout model labels them as page furniture, and they are excluded from its Markdown export. The plain-text fallback (PDFs over the Docling page cap, or that Docling fails on) keeps them. An earlier filter meant to catch them here (short blocks repeated 3+ times within one PDF) never took effect: it flagged blocks that the cleaner then re-scored and kept, and it only inflated the "blocks removed" audit count. It was deleted (2026-09-28) rather than switched on, because switching it on would remove content with no measurement behind it; that needs a before/after comparison on fallback PDFs first.

### Block Metrics & Removal Thresholds
Every block is evaluated using measurable signals:
- **Link Density**: The ratio of link characters to total text characters.
- **Word Count**: The absolute length of the block.
- **Document Frequency**: The fraction of the job's documents in which this block appears (counted only once at least `MIN_SHARING_DOCUMENTS` documents share it).

A block is removed if it passes an objective, data-driven threshold:
- High link density combined with low word count strongly indicates a navigation menu or footer.
- High document frequency combined with low word count strongly indicates repetitive boilerplate (e.g., copyright notices or site-wide banners).

### Scoring Weights and Removal Tiers (experiment)
Code blocks, tables and headings are never scored: they are always kept. Every other block gets a score from these signals, in `src/processing/cleaner.py`:

| Signal | Condition | Score |
|---|---|---|
| Frequency | document frequency > 80% / > 40% / > 10% | +5 / +3 / +1 |
| Edge position | in the first or last 10% of the document | +1.5 |
| Link density | > 0.8 / > 0.5 | +3 / +1 |
| Information density | under 10 words with a link / over 30 words with link density under 0.1 | +2 / -3 |
| Low diversity | over 5 words and unique-word ratio under 0.5 | +2 |
| Context | each neighbouring block that is short (under 15 words) and link-heavy (> 0.5) | +1 |

A block is removed only by one of three tiers, and only when its score is above -1:
1. **Obvious chrome:** document frequency above 95% and under 15 words.
2. **Multi-signal match:** score of at least 6 from at least two *different kinds* of signal (frequency, position, link density, information density, diversity, context).
3. **Link wall:** link density above 0.9 and document frequency above 20%.

*Status: experiment.* These weights and cut-offs were set by reasoning about what navigation chrome looks like, not fitted to a corpus or validated by an evaluation. The only evidence is unit tests on synthetic pages (a navigation block repeated on every page of a job is removed; the same block on a single page is kept) and the removal audit below. The intended failure mode is to keep noise: tier 2 needs two independent kinds of evidence, and tier 1 and 3 need cross-document repetition. The risk is the opposite on a corpus with unusual formatting, such as link-heavy reference tables written as plain lines. Before relying on them for a new corpus, read the removal audit for a sample of its documents, and measure retrieval with and without cleaning (Phase 6) before changing any value.

> **Corpus-Independence Rule:** No specific text, heading title, or keyword (e.g., "Related Links") is ever hardcoded as a removal trigger. Removal is always driven by statistical evidence from the corpus itself.

### Evidence Required Before Frequency Counts
Document frequency is only evidence when enough documents were compared. A block must appear in **at least 3 documents** (`MIN_SHARING_DOCUMENTS`) before its frequency is used at all. With two documents, any shared sentence has a frequency of 100%, and sharing a sentence between a pair is common for real content (a quoted definition, a licence line), so the ratio alone cannot separate chrome from coincidence. With fewer than three sharing documents, the block is kept. *Experiment:* three is the smallest count that is a pattern rather than a pair; it was set by reasoning, not tuned on a corpus. Measure removals per document on each new corpus before raising it.

### Shared Block Protection
Fenced code and tables are hidden behind placeholders while text is split by headings and blank lines, so a `#` inside a code block never reads as a heading and a blank line inside a fence never ends a block. One implementation (`src/protected_markdown.py`) serves both the cleaner and the chunker, so they cannot disagree about what a block is. Placeholders are delimited by NUL, which is stripped from the document first: no document can contain one, so text that merely looks like a placeholder is left alone. (Postgres cannot store NUL in text, so stripping it is required in any case.) The pattern for a table stops before the newline that ends its last row, which keeps a heading next to a table on its own line.

### Fake Headings
Bold-only lines are promoted to headings only in a document that has **no** real headings (outside code fences). Where the parser recovered a structure, a bold line is emphasis inside it, and promoting it would invent sections the author did not make.

### Pages Are Not Dropped For Being Short
A fetched page is rejected only when it has no words at all. Length is not evidence that content is useless, and a login wall or bot block that comes back for many URLs is recognised structurally instead: within one fetch job, a page whose text (ignoring whitespace) is identical to one already fetched is skipped and audited as `duplicate_content`. A single short page is kept.

### Inspecting What Was Removed
Every document logs its word count before and after cleaning and how many blocks were removed (`CLEAN | <url> | words 1200 -> 1105 | blocks removed 4/61`). At DEBUG level each removed block is logged with the signals that removed it and its first 80 characters, so a removal can always be traced to measured evidence. When cleaning leaves under 50 characters, the raw Markdown is chunked instead: that outcome is treated as a cleaner failure, not as an empty document.

### Content Preservation Philosophy
The overarching principle is: **when in doubt, preserve**. A small amount of noise retained is far less costly than accidentally removing genuine content. The thresholds above are intentionally conservative and biased toward false negatives (keeping unimportant text) rather than false positives (removing important text).

## Design Philosophy & Tradeoffs
- **Batch-Level vs. Stream-Level Analysis:** Corpus-frequency analysis requires seeing a sufficient batch of documents to identify what is truly "boilerplate" versus unique content. For single-document ingestions, the system gracefully falls back to relying primarily on structural signals (like link density) rather than cross-document frequency.
- **Single-Pass Cleaning:** The cleaner operates as a single-pass algorithm to maintain the high throughput required of a real-time microservice, opting against multi-pass iterative cleaning that would slow down ingestion.
