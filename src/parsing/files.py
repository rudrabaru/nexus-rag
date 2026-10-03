"""
Uploaded file -> Markdown.

- .txt / .md: read as text.
- .pdf / .docx: Docling (layout analysis, reading order, table structure, OCR of scanned pages
  with RapidOCR), in a child process.

Why a child process, and for every parser: the upload is untrusted, and the libraries that read
it (Docling, PyMuPDF, the zip reader) are large native code. Run in a fresh process they cannot
take the worker down with a crash or an out-of-memory kill, can be killed on a timeout, and hand
their memory back to the OS (Docling peaked at 1.7-3.4 GB per PDF in the 2026-09-26 spike,
docs/phases/phase1_ingestion.md, and does not release it between documents). The child answers
through a file, never a pickle (src/parsing/child.py).

Why one Docling at a time: a lock serialises it across the worker's concurrent jobs, so peak
memory is one document's, not WORKER_CONCURRENCY of them.

Fallback: a PDF over DOCLING_MAX_PAGES, or one Docling fails on (timeout, crash, OOM), is read
with PyMuPDF as plain text. That keeps the content but loses heading structure, and the
document is marked with the parser that produced it so the loss is visible.
"""
import logging
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings
from src.errors import UnprocessableSourceError
from src.parsing import child
from src.parsing.structure import promote_fake_headings

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md"}
DOCLING_EXTENSIONS = {".pdf", ".docx"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # the child is started as a module of this package
QUICK_PARSE_TIMEOUT_SECONDS = 120  # page count and plain text: seconds on any document the upload cap admits

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
        pages = int(_run_child("pages", path, QUICK_PARSE_TIMEOUT_SECONDS))
        if pages > settings.docling_max_pages:
            logger.warning(f"PARSE | {pages} pages > DOCLING_MAX_PAGES={settings.docling_max_pages}; using PyMuPDF text")
            return ParsedFile(_run_child("text", path, QUICK_PARSE_TIMEOUT_SECONDS), "pymupdf")

    try:
        markdown = _docling_markdown(path, settings.docling_timeout_seconds)
    except RuntimeError as e:
        if extension != ".pdf":
            raise UnprocessableSourceError(f"Could not parse the document: {e}") from e
        logger.warning(f"PARSE | Docling failed ({e}); using PyMuPDF text, which has no heading structure")
        return ParsedFile(_run_child("text", path, QUICK_PARSE_TIMEOUT_SECONDS), "pymupdf")

    if extension == ".docx":
        markdown = promote_fake_headings(markdown)
    return ParsedFile(markdown, "docling")


def _docling_markdown(path: str, timeout_seconds: int) -> str:
    with _one_docling_at_a_time:
        return _run_child("docling", path, timeout_seconds)


def _run_child(mode: str, path: str, timeout_seconds: int) -> str:
    """
    Runs one parser mode in a fresh process. Raises UnprocessableSourceError when the file itself is
    unusable, and RuntimeError on a failure of the parser (error, timeout, killed child).
    """
    with tempfile.TemporaryDirectory(prefix="nexus-parse-") as scratch:
        output = Path(scratch) / "out"
        try:
            done = subprocess.run(
                [sys.executable, "-m", "src.parsing.child", mode, path, str(output)],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_seconds, cwd=PROJECT_ROOT,
            )
        except subprocess.TimeoutExpired:  # the child is killed before this is raised
            raise RuntimeError(f"timed out after {timeout_seconds}s") from None
        if done.returncode == child.REJECTED:
            raise UnprocessableSourceError(done.stderr.strip() or "The file could not be parsed.")
        if done.returncode != child.DONE:
            detail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "no message"
            raise RuntimeError(f"the parser process failed (exit code {done.returncode}; {detail})")
        return output.read_text(encoding="utf-8")
