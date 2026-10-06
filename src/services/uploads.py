"""Accepting an uploaded file safely: a harmless label, an allowed type and a bounded read."""
import os
import re
from typing import Optional, Protocol

from src.services.errors import InvalidRequest, PayloadTooLarge

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
MAX_FILENAME_CHARS = 255
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class Upload(Protocol):
    """An uploaded file as the service needs it; FastAPI's UploadFile satisfies this."""

    filename: Optional[str]

    async def read(self, size: int = -1) -> bytes: ...


def safe_filename(raw: Optional[str]) -> str:
    """
    The label of an uploaded file: its last path segment under either separator style, without
    control characters. The worker may run on another OS than the API, so a name that is harmless
    here (a backslash path on Linux) must stay harmless there. It is a display label only: the
    worker never uses it as a path.
    """
    name = re.split(r"[\\/]", raw or "")[-1]
    name = "".join(c for c in name if c.isprintable()).strip()
    if not name or name in (".", "..") or len(name) > MAX_FILENAME_CHARS:
        raise InvalidRequest(f"The uploaded file needs a name of at most {MAX_FILENAME_CHARS} characters.")
    return name


def allowed_extension(filename: str) -> str:
    """The file's lower-case extension, or InvalidRequest when its type is not accepted."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise InvalidRequest(
            f"Unsupported file type: {ext}. Allowed types are: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
        )
    return ext


async def read_capped(file: Upload, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Reads an upload in chunks and stops as soon as it exceeds `limit`, so an oversized body is never held in memory."""
    chunks, total = [], 0
    while chunk := await file.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise PayloadTooLarge(f"File exceeds {limit // (1024 * 1024)}MB limit")
        chunks.append(chunk)
    return b"".join(chunks)
