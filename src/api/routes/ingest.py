import logging
from typing import Any, Optional
import asyncio

import procrastinate
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Depends

from src.api.rate_limit import INGEST_LIMIT, READ_LIMIT, limiter
from src.api.security import require_tenant
from src.api.dependencies import get_job_queue, get_registry, get_pipeline_logger
from src.api.models.ingest_models import JobStatusResponse
from src.registry.database import DocumentRegistry
from src.services.ingestion_service import prepare_and_queue_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest")
@limiter.limit(INGEST_LIMIT)
async def ingest_document(
    request: Request,
    url: Optional[str] = Form(None),
    file: UploadFile = File(None),
    tenant_id: str = Depends(require_tenant),
    resume: bool = Form(False),
    registry: DocumentRegistry = Depends(get_registry),
    job_queue: procrastinate.App = Depends(get_job_queue),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    response = await prepare_and_queue_ingestion(job_queue, registry, tenant_id, url, file, resume)

    if pipeline_logger:
        pipeline_logger.log_event(
            "ingestion_queued", job_id=response["job_id"], tenant_id=tenant_id, source=url or (file.filename if file else None)
        )

    return response


@router.get("/ingest/{job_id}", response_model=JobStatusResponse)
@limiter.limit(READ_LIMIT)
async def get_job_status(
    request: Request,
    job_id: str,
    tenant_id: str = Depends(require_tenant),
    registry: DocumentRegistry = Depends(get_registry),
):
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
