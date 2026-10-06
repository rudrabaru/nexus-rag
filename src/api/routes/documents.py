import asyncio
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from src.api.dependencies import get_documents, get_ingestion_service, get_jobs, get_pipeline_logger
from src.api.rate_limit import INGEST_LIMIT, READ_LIMIT, limiter
from src.api.schemas.documents import DeleteDocumentResponse, DocumentResponse, IngestResponse, JobStatusResponse
from src.api.security import Principal, require_principal, require_tenant
from src.services.ingestion_service import IngestionService
from src.stores.documents import DocumentStore
from src.stores.jobs import JobStore

router = APIRouter(prefix="/v1", tags=["documents"])


def list_for(documents: DocumentStore, principal: Principal) -> list:
    return documents.list_all_documents() if principal.is_admin else documents.list_documents(principal.tenant_id)


@router.post("/documents", response_model=IngestResponse, status_code=202, operation_id="add_document")
@limiter.limit(INGEST_LIMIT)
async def add_document(
    request: Request,
    url: Optional[str] = Form(None, description="A web page or sitemap URL (https)."),
    file: UploadFile = File(None, description="A PDF, DOCX, TXT or MD file, up to 20 MB."),
    resume: bool = Form(False, description="Only add what is missing from an earlier partial run."),
    tenant_id: str = Depends(require_tenant),
    service: IngestionService = Depends(get_ingestion_service),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    """Queues a source for ingestion. Processing happens in a worker; poll `/v1/jobs/{job_id}`."""
    submission = await service.submit(tenant_id, url, file, resume)
    if pipeline_logger:
        pipeline_logger.log_event(
            "ingestion_queued", job_id=submission.job_id, tenant_id=tenant_id, source=url or (file.filename if file else None)
        )
    return IngestResponse(job_id=submission.job_id, status=submission.status, warning=submission.warning)


@router.get("/documents", response_model=List[DocumentResponse], operation_id="list_documents")
@limiter.limit(READ_LIMIT)
async def list_documents(
    request: Request,
    documents: DocumentStore = Depends(get_documents),
    principal: Principal = Depends(require_principal),
):
    """The documents of the caller's workspace (every workspace for the admin)."""
    docs = await asyncio.to_thread(list_for, documents, principal)
    return [
        DocumentResponse(
            id=d["doc_id"],
            url=d["source"],
            title=d["source"].split("/")[-1] if "/" in d["source"] else d["source"],
            status=d["status"],
            created_at=d["ingested_at"],
            chunks=d["chunk_count"],
            error=d.get("error"),
            stats=d.get("stats") or {},
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse, operation_id="delete_document")
@limiter.limit(READ_LIMIT)
async def delete_document(
    request: Request,
    doc_id: str,
    documents: DocumentStore = Depends(get_documents),
    principal: Principal = Depends(require_principal),
):
    """Deletes a document. Its chunks, vectors and keyword-index entries go with it in one transaction."""
    doc = await asyncio.to_thread(documents.get_document, doc_id)
    # Another workspace's document is reported as missing, so document ids cannot be probed.
    if not doc or (not principal.is_admin and doc.get("tenant_id") != principal.tenant_id):
        raise HTTPException(status_code=404, detail="Document not found")

    await asyncio.to_thread(documents.delete_document, doc_id)
    return DeleteDocumentResponse(status="success", message=f"Deleted document {doc_id} and all related chunks.")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, operation_id="get_job", tags=["jobs"])
@limiter.limit(READ_LIMIT)
async def get_job(
    request: Request,
    job_id: str,
    tenant_id: str = Depends(require_tenant),
    jobs: JobStore = Depends(get_jobs),
    documents: DocumentStore = Depends(get_documents),
):
    """Progress of an ingestion job."""
    job = await asyncio.to_thread(jobs.get_job, job_id)
    doc = await asyncio.to_thread(documents.get_document, job["doc_id"]) if job else None
    # A job owned by another tenant is reported as missing so job IDs cannot be probed.
    if not job or not doc or doc.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobStatusResponse(
        job_id=job["job_id"], status=job["status"], progress_pct=job["progress_pct"], error=job["error"],
        doc_id=job["doc_id"], metadata=job.get("metadata"),
    )
    if job["status"] in ("complete", "partial_success"):
        response.chunk_count = doc["chunk_count"]
    return response
