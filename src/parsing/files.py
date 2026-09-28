"""
Uploaded file -> Markdown.

- .txt / .md: read as text.
- .pdf / .docx: Docling (layout analysis, reading order, table structure, OCR of scanned pages
  with RapidOCR), in a child process.

Why a child process per document: Docling's peak memory was 1.7-3.4 GB per PDF in the
2026-09-26 spike (docs/phases/phase1_ingestion.md), and its memory is not reliably released
between documents. A fresh process returns it to the OS, can be killed on a timeout, and an
out-of-memory kill takes down only the child.

Why one at a time: a lock serialises Docling across the worker's concurrent jobs, so peak
memory is one document's, not WORKER_CONCURRENCY of them.

Fallback: a PDF over DOCLING_MAX_PAGES, or one Docling fails on (timeout, crash, OOM), is read
with PyMuPDF as plain text. That keeps the content but loses heading structure, and the
document is marked with the parser that produced it so the loss is visible.
"""
import logging
import multiprocessing
import threading
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.config import get_settings
from src.ingestion.errors import UnprocessableSourceError
from src.parsing import docling_child
from src.parsing.structure import promote_fake_headings

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md"}
DOCLING_EXTENSIONS = {".pdf", ".docx"}

_one_docling_at_a_time = threading.Lock()


@dataclass
class ParsedFile:
    markdown: str
    parser: str  # text | docling | pymupdf


def parse_file(path: str) -> ParsedFile:
    extension = Path(path).suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return ParsedFile(Path(path).read_text(encoding="utf-8", errors="replace"), "text")
    if extension not in DOCLING_EXTENSIONS:
        raise UnprocessableSourceError(f"Unsupported file type: {extension}")

    settings = get_settings()
    if extension == ".pdf":
        pages = _pdf_page_count(path)
        if pages > settings.docling_max_pages:
            logger.warning(f"PARSE | {pages} pages > DOCLING_MAX_PAGES={settings.docling_max_pages}; using PyMuPDF text")
            return ParsedFile(_pymupdf_text(path), "pymupdf")

    try:
        markdown = _docling_markdown(path, settings.docling_timeout_seconds)
    except RuntimeError as e:
        if extension != ".pdf":
            raise UnprocessableSourceError(f"Could not parse the document: {e}") from e
        logger.warning(f"PARSE | Docling failed ({e}); using PyMuPDF text, which has no heading structure")
        return ParsedFile(_pymupdf_text(path), "pymupdf")

    if extension == ".docx":
        markdown = promote_fake_headings(markdown)
    return ParsedFile(markdown, "docling")


def _docling_markdown(path: str, timeout_seconds: int) -> str:
    """Runs Docling in a fresh spawned process. Raises RuntimeError on error, timeout or a killed child."""
    context = multiprocessing.get_context("spawn")
    with _one_docling_at_a_time:
        receiver, sender = context.Pipe(duplex=False)
        child = context.Process(target=docling_child.convert, args=(path, sender), daemon=True)
        child.start()
        sender.close()  # the parent's copy: EOF on the receiver once the child is gone
        try:
            if not receiver.poll(timeout_seconds):
                raise RuntimeError(f"timed out after {timeout_seconds}s")
            status, payload = receiver.recv()
        except EOFError:
            child.join(timeout=5)
            raise RuntimeError(f"the parser process died (exit code {child.exitcode}; out of memory?)")
        finally:
            if child.is_alive():
                child.kill()
            child.join(timeout=5)
            receiver.close()
    if status != "ok":
        raise RuntimeError(payload)
    return payload


def _pdf_page_count(path: str) -> int:
    try:
        with pymupdf.open(path) as pdf:
            return pdf.page_count
    except Exception as e:
        raise UnprocessableSourceError(f"The PDF could not be opened: {e}") from e


def _pymupdf_text(path: str) -> str:
    with pymupdf.open(path) as pdf:
        return "\n\n".join(page.get_text() for page in pdf)
