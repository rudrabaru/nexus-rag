"""
Parsing: turn an uploaded file into Markdown that keeps its structure (headings, tables).

Worker-only (Dockerfile.worker): Docling brings torch and layout models, which the API image
must never carry. Web pages arrive as Markdown from reader APIs and are not parsed here.
"""
