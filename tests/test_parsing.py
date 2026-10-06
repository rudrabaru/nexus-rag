"""Upload parsing: text passthrough, Docling in a child process, and the PyMuPDF fallbacks."""
import os
import zipfile

import pymupdf
import pytest

from src.config import get_settings
from src.errors import UnprocessableSourceError
from src.parsing import child, files
from src.parsing.files import parse_file
from src.parsing.structure import promote_fake_headings


def make_pdf(path, pages=2, text="Load balancing distributes traffic across healthy backends."):
    with pymupdf.open() as pdf:
        for i in range(pages):
            page = pdf.new_page()
            page.insert_text((72, 72), f"Section {i + 1}", fontsize=20)
            page.insert_text((72, 110), text, fontsize=11)
        pdf.save(path)
    return str(path)


def test_text_and_markdown_are_read_as_is(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nBody", encoding="utf-8")
    parsed = parse_file(str(tmp_path / "a.md"))
    assert parsed.markdown == "# Title\n\nBody" and parsed.parser == "text"


def test_unsupported_types_are_unprocessable(tmp_path):
    (tmp_path / "a.exe").write_bytes(b"MZ")
    with pytest.raises(UnprocessableSourceError):
        parse_file(str(tmp_path / "a.exe"))


def test_a_pdf_over_the_page_cap_is_read_as_text_without_docling(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCLING_MAX_PAGES", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(files, "_docling_markdown", lambda *a: pytest.fail("Docling must not run"))
    parsed = parse_file(make_pdf(tmp_path / "long.pdf", pages=2))
    assert parsed.parser == "pymupdf" and "Load balancing" in parsed.markdown


def test_a_pdf_docling_fails_on_keeps_its_content_via_pymupdf(tmp_path, monkeypatch):
    def fail(path, timeout):
        raise RuntimeError("the parser process died (exit code -9; out of memory?)")

    monkeypatch.setattr(files, "_docling_markdown", fail)
    parsed = parse_file(make_pdf(tmp_path / "a.pdf"))
    assert parsed.parser == "pymupdf" and "Load balancing" in parsed.markdown


def test_a_docx_docling_fails_on_is_unprocessable(tmp_path, monkeypatch):
    (tmp_path / "a.docx").write_bytes(b"not really a docx")
    monkeypatch.setattr(files, "_docling_markdown", lambda path, timeout: (_ for _ in ()).throw(RuntimeError("bad file")))
    with pytest.raises(UnprocessableSourceError):
        parse_file(str(tmp_path / "a.docx"))


def test_an_unreadable_pdf_is_unprocessable(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(UnprocessableSourceError):
        parse_file(str(tmp_path / "broken.pdf"))


def test_a_child_that_overruns_its_timeout_is_killed(tmp_path):
    """Real spawned child: one second is far below Docling's model load, so the timeout path runs."""
    with pytest.raises(RuntimeError, match="timed out"):
        files._docling_markdown(make_pdf(tmp_path / "a.pdf"), timeout_seconds=1)


def test_bold_only_lines_become_headings_but_sentences_do_not():
    """Regression: the punctuation check read the raw line, which always ends in '**', so it never excluded anything."""
    markdown = "**Setup Guide**\n\nText.\n\n**Loop & Wait Nodes:**\n\nText.\n\n**This is a sentence.**\n\n**Inline** bold text."
    promoted = promote_fake_headings(markdown)
    assert "## Setup Guide" in promoted and "## Loop & Wait Nodes:" in promoted
    assert "**This is a sentence.**" in promoted and "**Inline** bold text." in promoted


@pytest.mark.skipif(os.environ.get("RUN_DOCLING_TESTS") != "1", reason="loads Docling's models; set RUN_DOCLING_TESTS=1")
def test_docling_parses_a_pdf_in_a_child_process(tmp_path):
    parsed = parse_file(make_pdf(tmp_path / "a.pdf"))
    assert parsed.parser == "docling"
    assert "Load balancing distributes traffic" in parsed.markdown
    assert "&amp;" not in parsed.markdown and "<!-- image -->" not in parsed.markdown


# ── The child process: untrusted input stays out of the worker ──────────────

def test_the_page_count_comes_from_the_child_process(tmp_path):
    assert files._run_child("pages", make_pdf(tmp_path / "a.pdf", pages=3), 60) == "3"


def test_an_unusable_file_is_rejected_not_reported_as_a_parser_failure(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(UnprocessableSourceError):
        files._run_child("pages", str(tmp_path / "broken.pdf"), 60)


def test_a_crashing_parser_is_a_runtime_error_with_its_reason(tmp_path):
    with pytest.raises(RuntimeError, match="parser process failed"):
        files._run_child("text", str(tmp_path / "missing.pdf"), 60)


def test_a_docx_that_unpacks_far_beyond_its_size_is_refused(tmp_path, monkeypatch):
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * (3 * 1024 * 1024))
    monkeypatch.setattr(child, "MAX_UNPACKED_DOCX_BYTES", 1024 * 1024)
    with pytest.raises(child.RejectedInput, match="unpacks to"):
        child.check_unpacked_size(str(bomb))


def test_a_docx_that_is_not_a_zip_is_refused(tmp_path):
    (tmp_path / "fake.docx").write_bytes(b"not a zip")
    with pytest.raises(child.RejectedInput, match="not a valid DOCX"):
        child.check_unpacked_size(str(tmp_path / "fake.docx"))
