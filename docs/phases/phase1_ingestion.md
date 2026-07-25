# Phase 1: Ingestion

## Overview
The ingestion phase serves as the entry point for all raw data entering the Retrieval-Augmented Generation (RAG) system. The core objective is to ingest heterogeneous data formats and normalize them into a uniform, text-based intermediate representation (Markdown) while explicitly preserving structural hierarchies and metadata.

## Core Implementation Logic

### Format Routing
An intelligent routing engine dynamically directs incoming sources to the appropriate processing pipeline:
- **Web URLs**: The router first detects the actual content type. If the URL resolves to a binary document (like a PDF or Word file), it is downloaded and routed to the document extraction engine. Standard web pages are routed to a headless browser crawler.
- **File Uploads**: Files are routed based on their format directly to the document extraction engine.

### Multi-Format Extraction Engine
A unified parsing system converts virtually any document format directly into clean Markdown.

1. **Standard Extraction:** The engine reads text, tables, and lists natively from office documents and PDFs, converting them into structured Markdown.
2. **Multimodal Vision Extraction (Optional):** When requested, a vision-capable AI model intercepts images and scanned pages within the documents. It interprets the visual content and injects text descriptions directly into the Markdown, expanding retrieval capabilities to scanned PDFs and image-heavy presentations.
3. **Structural Post-processing:** The engine automatically recovers structural hierarchy from documents that use typographic conventions instead of native heading styles. It also sanitizes artifacts like tracked changes.

### Web Crawling Engine
For dynamic HTML pages, the system employs a headless browser capable of rendering JavaScript. This ensures that content hidden behind dynamic loads or single-page applications is fully captured and converted to structured Markdown.

### In-Memory Processing & Multi-Tenancy
The entire ingestion process operates as an in-memory microservice. Temporary files created during uploads are stored in the OS temp directory and are deleted immediately after parsing. No raw source files are permanently stored on disk.

All documents are tagged at the ingestion layer with:
- **Tenant ID**: Identifies the owning workspace to ensure strict data isolation.
- **Visibility**: Defines the scope of the document (e.g., restricted strictly to the owning tenant).

### Deduplication
Before starting an extraction job, the system computes a cryptographic hash of the raw file content. If an identical hash already exists in the registry for the same tenant with a completed status, the ingestion is skipped. This prevents redundant reprocessing of identical documents.

## Design Philosophy & Tradeoffs
- **Simplicity vs. Fidelity:** The unified extraction approach favors a fast, uniform conversion layer over highly specialized format parsers. While extremely complex academic layouts might lose some visual context, it significantly reduces pipeline maintenance overhead.
- **Crawler Reliability:** JavaScript-heavy sites behind aggressive bot detection may fail to render fully. In these edge cases, the system relies on manual document fallback rather than building brittle, site-specific scrapers.
