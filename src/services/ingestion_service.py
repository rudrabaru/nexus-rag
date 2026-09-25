"""
Validates an ingestion request and hands it to the durable job queue.

Nothing here parses a document: fetching, chunking and embedding happen in the worker
(src/jobs/tasks.py), so this module carries no document-parsing dependencies and the API
image stays slim. An uploaded file's bytes are read into memory, hashed, and stored in the
`ingest_sources` table in the same transaction as the job — never on the API's local disk,
which the worker (possibly a different host) cannot see.
"""
import asyncio
import hashlib
import os
import uuid
from typing import Optional

import procrastinate
from fastapi import HTTPException, UploadFile

from src.ingestion.url_policy import UnsafeUrlError, validate_public_url
from src.jobs.contract import INGEST_QUEUE, INGEST_TASK, IngestionRequest
from src.registry.database import DocumentRegistry

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Bounds how much of Neon's 0.5 GB free-tier storage (7a) an offline or backlogged worker
# can consume with unprocessed uploads: 200 MB leaves the rest for chunk vectors. An
# operational safety limit on ingest_sources, not a corpus-tuned retrieval threshold.
MAX_PENDING_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_SITEMAP_ESTIMATE_CHUNKS = 500
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
TENANT_CHUNK_QUOTA = 2000


def _doc_id(tenant_id: str, source_ident: str) -> str:
    return hashlib.blake2b(f"{tenant_id}_{source_ident}".encode(), digest_size=16).hexdigest()


async def _check_quota(registry: DocumentRegistry, tenant_id: str) -> None:
    quota = await asyncio.to_thread(registry.get_tenant_quota, tenant_id)
    if quota >= TENANT_CHUNK_QUOTA:
        raise HTTPException(
            status_code=429,
            detail=f"Tenant quota exceeded ({TENANT_CHUNK_QUOTA} chunks max). Delete old documents to ingest new ones.",
        )


async def _defer_job(job_queue: procrastinate.App, doc_id: str, request: IngestionRequest) -> None:
    """
    lock=doc_id serialises concurrent jobs for the same document (e.g. a resume retried
    while the original run is still in flight), so two workers never write the same
    document's chunks at once.
    """
    await asyncio.to_thread(
        lambda: job_queue.configure_task(INGEST_TASK, queue=INGEST_QUEUE, lock=doc_id).defer(**request.model_dump())
    )


async def prepare_and_queue_ingestion(
    job_queue: procrastinate.App,
    registry: DocumentRegistry,
    tenant_id: str,
    url: Optional[str],
    file: Optional[UploadFile],
    extract_visuals: bool,
    resume: bool,
) -> dict:
    """Validates the request, registers the job durably, and defers it to the worker queue."""
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide either url or file")

    if url:
        try:
            await asyncio.to_thread(validate_public_url, url)
        except UnsafeUrlError as e:
            raise HTTPException(status_code=400, detail=str(e))

    await _check_quota(registry, tenant_id)

    filename = None
    content_hash = None
    upload = None
    total_size = 0

    if url:
        doc_id = _doc_id(tenant_id, url)
        format_type = "sitemap" if (url.lower().endswith(".xml") or "sitemap" in url.lower()) else "web"
    else:
        filename = os.path.basename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed types are: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
            )

        pending = await asyncio.to_thread(registry.pending_upload_bytes, tenant_id)
        if pending >= MAX_PENDING_UPLOAD_BYTES:
            raise HTTPException(
                status_code=429, detail="Too many documents already queued for processing. Wait for them to finish."
            )

        content = await file.read()
        total_size = len(content)
        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")

        content_hash = hashlib.sha256(content).hexdigest()
        upload = (filename, content)
        format_type = ext.lstrip(".")
        doc_id = _doc_id(tenant_id, filename)

        existing = await asyncio.to_thread(registry.get_document_by_hash, tenant_id, content_hash)
        if existing and existing["status"] == "complete":
            dummy_job_id = str(uuid.uuid4())
            await asyncio.to_thread(
                registry.register_job, dummy_job_id, existing["doc_id"], existing["source"],
                existing["format"], tenant_id, content_hash,
            )
            await asyncio.to_thread(registry.update_job_status, dummy_job_id, "complete", 100)
            return {"job_id": dummy_job_id, "status": "complete"}

    if not resume:
        # Re-ingesting replaces the document; its chunks and vectors cascade with it.
        await asyncio.to_thread(registry.delete_document, doc_id)

    job_id = str(uuid.uuid4())
    request = IngestionRequest(
        job_id=job_id, doc_id=doc_id, tenant_id=tenant_id, url=url, filename=filename,
        content_hash=content_hash, extract_visuals=extract_visuals, resume=resume,
    )
    await asyncio.to_thread(
        registry.register_job, job_id, doc_id, request.source_ref, format_type, tenant_id, content_hash, upload=upload,
    )
    await _defer_job(job_queue, doc_id, request)

    response = {"job_id": job_id, "status": "queued"}
    if format_type == "sitemap":
        response["warning"] = "Sitemap detected. Pages will be crawled sequentially."
    elif total_size and total_size / 2000 > MAX_SITEMAP_ESTIMATE_CHUNKS:
        response["warning"] = (
            f"Large document (~{int(total_size / 2000)} estimated chunks). "
            "Ingestion may take several minutes; check progress with GET /ingest/{job_id}."
        )
    return response
