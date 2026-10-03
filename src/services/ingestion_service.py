"""
Validates an ingestion request and hands it to the durable job queue.

Nothing here parses a document: fetching, chunking and embedding happen in the worker
(src/jobs/ingest_tasks.py), so this module carries no document-parsing dependencies and the API
image stays slim. An uploaded file's bytes are read into memory (in bounded chunks), hashed, and
stored in the `ingest_sources` table in the same transaction as the job: never on the API's local
disk, which the worker (possibly a different host) cannot see.

Re-ingesting a document does not delete it. The old chunks keep serving queries until the new run
commits and replaces them in one transaction, so a worker that is offline for days costs nothing.
"""
import asyncio
import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol

import procrastinate

from src.config import Settings, get_settings
from src.crawling.policy import check_fetchable
from src.crawling.sitemap import MAX_SITEMAP_PAGES, is_sitemap_url
from src.ingestion.url_policy import UnsafeUrlError
from src.jobs.contract import FETCH_QUEUE, FETCH_TASK, INGEST_QUEUE, INGEST_TASK, IngestionRequest
from src.services.errors import InvalidRequest, PayloadTooLarge, QuotaExceeded, Unavailable
from src.stores.documents import DocumentStore
from src.stores.fetches import FetchStore
from src.stores.jobs import JobStore

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
MAX_FILENAME_CHARS = 255
# Bounds how much of Neon's 1 GB free-tier storage an offline or backlogged worker can consume with
# unprocessed uploads: 200 MB leaves the rest for chunk vectors. An operational safety limit on
# ingest_sources, not a corpus-tuned retrieval threshold.
MAX_PENDING_UPLOAD_BYTES = 200 * 1024 * 1024
# Jobs a tenant may have queued or running at once. Workers run on demand, so work can wait for
# days; this bounds how much one tenant can pile up in that time. A limit chosen to be generous for
# a person uploading by hand, not tuned to any corpus.
MAX_ACTIVE_JOBS_PER_TENANT = 10
TENANT_CHUNK_QUOTA = 2000
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
# A document that will make about this many chunks gets a heads-up that it will take a while.
LARGE_DOCUMENT_WARNING_CHUNKS = 500
ESTIMATED_BYTES_PER_CHUNK = 2000


class Upload(Protocol):
    """An uploaded file as the service needs it; FastAPI's UploadFile satisfies this."""

    filename: Optional[str]

    async def read(self, size: int = -1) -> bytes: ...


@dataclass
class Submission:
    job_id: str
    status: str  # queued | complete (the content was already indexed)
    warning: Optional[str] = None


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


async def _read_capped(file: Upload, limit: int) -> bytes:
    """Reads an upload in chunks and stops as soon as it exceeds `limit`, so an oversized body is never held in memory."""
    chunks, total = [], 0
    while chunk := await file.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise PayloadTooLarge(f"File exceeds {limit // (1024 * 1024)}MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _doc_id(tenant_id: str, source_ident: str) -> str:
    return hashlib.blake2b(f"{tenant_id}_{source_ident}".encode(), digest_size=16).hexdigest()


class IngestionService:
    def __init__(
        self, job_queue: procrastinate.App, documents: DocumentStore, jobs: JobStore, fetches: FetchStore,
        settings: Optional[Settings] = None,
    ):
        self._queue = job_queue
        self._documents = documents
        self._jobs = jobs
        self._fetches = fetches
        self._settings = settings or get_settings()

    async def submit(self, tenant_id: str, url: Optional[str], file: Optional[Upload], resume: bool) -> Submission:
        """Validates the request, registers the job durably, and defers it to the worker queue."""
        if not url and not file:
            raise InvalidRequest("Must provide either url or file")

        if url:
            await self._check_url(tenant_id, url)
        await self._check_quota(tenant_id)

        content_hash, upload, size = None, None, 0
        if url:
            doc_id = _doc_id(tenant_id, url)
            format_type = "sitemap" if is_sitemap_url(url) else "web"
            filename = None
        else:
            filename = safe_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_UPLOAD_EXTENSIONS:
                raise InvalidRequest(
                    f"Unsupported file type: {ext}. Allowed types are: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
                )
            if await asyncio.to_thread(self._jobs.pending_upload_bytes, tenant_id) >= MAX_PENDING_UPLOAD_BYTES:
                raise QuotaExceeded("Too many documents already queued for processing. Wait for them to finish.")

            content = await _read_capped(file, MAX_UPLOAD_BYTES)
            size = len(content)
            content_hash = hashlib.sha256(content).hexdigest()
            upload, format_type, doc_id = (filename, content), ext.lstrip("."), _doc_id(tenant_id, filename)

            existing = await asyncio.to_thread(self._documents.get_document_by_hash, tenant_id, content_hash)
            if existing and existing["status"] == "complete":
                return await self._already_indexed(tenant_id, existing, content_hash)

        job_id = str(uuid.uuid4())
        request = IngestionRequest(
            job_id=job_id, doc_id=doc_id, tenant_id=tenant_id, url=url, filename=filename,
            content_hash=content_hash, resume=resume,
        )
        await asyncio.to_thread(
            self._jobs.register_job, job_id, doc_id, request.source_ref, format_type, tenant_id, content_hash,
            upload=upload,
        )
        await self._defer(job_id, doc_id, request)
        return Submission(job_id, "queued", self._warning(format_type, size))

    async def _check_url(self, tenant_id: str, url: str) -> None:
        """The fetch policy (https, public address, domain lists) and the tenant's daily page quota."""
        settings = self._settings
        try:
            await asyncio.to_thread(check_fetchable, url, settings.allowed_fetch_domains, settings.denied_fetch_domains)
        except UnsafeUrlError as e:
            raise InvalidRequest(str(e))
        if await asyncio.to_thread(self._fetches.pages_fetched_today, tenant_id) >= settings.fetch_daily_page_quota:
            raise QuotaExceeded(
                f"Daily web page quota reached ({settings.fetch_daily_page_quota} pages per 24 hours). "
                "Upload files instead, or try later."
            )

    async def _check_quota(self, tenant_id: str) -> None:
        if await asyncio.to_thread(self._documents.chunk_count, tenant_id) >= TENANT_CHUNK_QUOTA:
            raise QuotaExceeded(
                f"Tenant quota exceeded ({TENANT_CHUNK_QUOTA} chunks max). Delete old documents to ingest new ones."
            )
        if await asyncio.to_thread(self._jobs.active_job_count, tenant_id) >= MAX_ACTIVE_JOBS_PER_TENANT:
            raise QuotaExceeded(
                f"{MAX_ACTIVE_JOBS_PER_TENANT} documents are already queued or processing. Wait for some to finish."
            )

    async def _already_indexed(self, tenant_id: str, existing: dict, content_hash: str) -> Submission:
        """The same bytes are already a complete document: record a finished job pointing at it, queue nothing."""
        job_id = str(uuid.uuid4())
        await asyncio.to_thread(
            self._jobs.register_job, job_id, existing["doc_id"], existing["source"], existing["format"], tenant_id, content_hash
        )
        await asyncio.to_thread(self._jobs.update_job_status, job_id, "complete", 100)
        return Submission(job_id, "complete")

    async def _defer(self, job_id: str, doc_id: str, request: IngestionRequest) -> None:
        """
        A URL goes to the fetch queue first (the fetch worker then defers ingest); an upload goes
        straight to ingest. lock=doc_id serialises concurrent jobs for the same document (e.g. a
        resume retried while the original run is still in flight), so two workers never write the
        same document's chunks at once. If the queue cannot take the job, the registered job is
        failed and its upload discarded, so no queued job exists that no worker will ever see.
        """
        task, queue = (FETCH_TASK, FETCH_QUEUE) if request.url else (INGEST_TASK, INGEST_QUEUE)
        try:
            await asyncio.to_thread(
                lambda: self._queue.configure_task(task, queue=queue, lock=doc_id).defer(**request.model_dump())
            )
        except Exception:
            logger.exception(f"Could not defer job {job_id}")
            await asyncio.to_thread(self._jobs.fail_job, job_id, "The job could not be queued.")
            raise Unavailable("The job queue is unavailable right now. Try again shortly.")

    @staticmethod
    def _warning(format_type: str, size: int) -> Optional[str]:
        if format_type == "sitemap":
            return f"Sitemap detected. Up to {MAX_SITEMAP_PAGES} pages are fetched one at a time through a reader API."
        if size and size / ESTIMATED_BYTES_PER_CHUNK > LARGE_DOCUMENT_WARNING_CHUNKS:
            return (
                f"Large document (~{int(size / ESTIMATED_BYTES_PER_CHUNK)} estimated chunks). "
                "Ingestion may take a while; check progress with GET /ingest/{job_id}."
            )
        return None
