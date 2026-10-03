"""Uploaded files: the name is a label (never a path) and the body is read with a hard cap."""
import os
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.jobs import ingest_tasks as tasks
from src.jobs.contract import IngestionRequest
from src.services.ingestion_service import MAX_UPLOAD_BYTES, READ_CHUNK_BYTES, prepare_and_queue_ingestion, safe_filename


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("..\\..\\Users\\x\\notes.md", "notes.md"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\x\\thesis.docx", "thesis.docx"),
        ("D:/Docs/thesis.docx", "thesis.docx"),
        ("na\x00me.md", "name.md"),
    ],
)
def test_the_stored_filename_is_a_single_path_segment(raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "..", "dir/", "a" * 300 + ".md"])
def test_unusable_filenames_are_rejected(raw):
    with pytest.raises(HTTPException) as exc:
        safe_filename(raw)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("stored_name", ["..\\..\\Users\\x\\notes.md", "C:\\Users\\x\\notes.md", "../../notes.md"])
def test_the_worker_writes_uploads_inside_its_temp_directory_under_a_fixed_name(stored_name, monkeypatch):
    """A name that was harmless on the API's OS must not escape the temp directory on the worker's."""
    seen = {}

    def fake_parse(path):
        seen["path"] = path
        seen["written"] = open(path, "rb").read()
        return MagicMock(markdown="# hi", parser="text")

    monkeypatch.setattr(tasks, "parse_file", fake_parse)
    registry = MagicMock()
    registry.get_ingest_source.return_value = (stored_name, b"# hi")
    request = IngestionRequest(job_id="j", doc_id="d", tenant_id="t", filename=stored_name)

    tasks._uploaded_document(request, registry, MagicMock())

    assert os.path.basename(seen["path"]) == "upload.md"
    assert os.path.basename(os.path.dirname(seen["path"])).startswith("nexus-ingest-")
    assert seen["written"] == b"# hi"


class CountingUpload:
    def __init__(self, filename: str, size: int):
        self.filename = filename
        self.remaining = size
        self.bytes_read = 0

    async def read(self, n: int = -1) -> bytes:
        n = self.remaining if n < 0 else min(n, self.remaining)
        self.remaining -= n
        self.bytes_read += n
        return b"x" * n


def make_registry():
    registry = MagicMock()
    registry.get_tenant_quota.return_value = 0
    registry.pending_upload_bytes.return_value = 0
    registry.get_document_by_hash.return_value = None
    return registry


@pytest.mark.asyncio
async def test_an_oversized_upload_is_refused_with_413_after_reading_at_most_one_extra_chunk():
    upload = CountingUpload("big.txt", MAX_UPLOAD_BYTES * 5)
    with pytest.raises(HTTPException) as exc:
        await prepare_and_queue_ingestion(MagicMock(), make_registry(), "tenant-1", None, upload, False)

    assert exc.value.status_code == 413
    assert upload.bytes_read <= MAX_UPLOAD_BYTES + READ_CHUNK_BYTES


@pytest.mark.asyncio
async def test_an_upload_exactly_at_the_limit_is_accepted():
    registry = make_registry()
    await prepare_and_queue_ingestion(MagicMock(), registry, "tenant-1", None, CountingUpload("ok.txt", MAX_UPLOAD_BYTES), False)
    registry.register_job.assert_called_once()
