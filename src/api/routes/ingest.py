import logging
from typing import Any, Optional
import asyncio

import procrastinate
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Depends
from slowapi import Limiter

from src.api.auth import get_current_tenant, get_rate_limit_key
from src.api.dependencies import get_job_queue, get_registry, get_pipeline_logger
from src.api.models.ingest_models import JobStatusResponse
from src.registry.database import DocumentRegistry
from src.services.ingestion_service import prepare_and_queue_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_rate_limit_key)


@router.post("/ingest")
@limiter.limit("10/minute")
async def ingest_document(
    request: Request,
    url: Optional[str] = Form(None),
    file: UploadFile = File(None),
    tenant_id: Optional[str] = Depends(get_current_tenant),
    extract_visuals: bool = Form(False),
    resume: bool = Form(False),
    registry: DocumentRegistry = Depends(get_registry),
    job_queue: procrastinate.App = Depends(get_job_queue),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    if not tenant_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a valid API key via the X-API-Key header to upload documents.",
        )

    response = await prepare_and_queue_ingestion(job_queue, registry, tenant_id, url, file, extract_visuals, resume)

    if pipeline_logger:
        pipeline_logger.log_event(
            "ingestion_queued", job_id=response["job_id"], tenant_id=tenant_id, source=url or (file.filename if file else None)
        )

    return response


@router.get("/ingest/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    registry: DocumentRegistry = Depends(get_registry)
):
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    job = await asyncio.to_thread(registry.get_job, job_id)
    doc = await asyncio.to_thread(registry.get_document, job["doc_id"]) if job else None
    # A job owned by another tenant is reported as missing so job IDs cannot be probed.
    if not job or not doc or doc.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_pct=job["progress_pct"],
        error=job["error"],
        doc_id=job["doc_id"],
        metadata=job.get("metadata"),
    )

    if job["status"] in ("complete", "partial_success"):
        response.chunk_count = doc["chunk_count"]

    return response
