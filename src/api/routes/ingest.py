import shutil
import logging
from typing import Optional
import uuid
import asyncio
import json
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form,
    HTTPException,
    BackgroundTasks,
    Request,
    Depends,
)
from src.api.auth import get_current_tenant
from slowapi import Limiter
from src.api.auth import get_real_ip
from src.api.dependencies import get_registry, get_retriever, get_ingestion_semaphore, get_pipeline_logger
from src.registry.database import DocumentRegistry
from typing import Any
from src.api.models.ingest_models import JobStatusResponse
from src.services.ingestion_service import _process_ingestion, prepare_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_real_ip)

@router.post("/ingest")
@limiter.limit("30/minute")
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    file: UploadFile = File(None),
    tenant_id: Optional[str] = Depends(get_current_tenant),
    extract_visuals: bool = Form(False),
    registry: DocumentRegistry = Depends(get_registry),
    semaphore: asyncio.Semaphore = Depends(get_ingestion_semaphore),
    retriever: Any = Depends(get_retriever),
    pipeline_logger: Any = Depends(get_pipeline_logger),
):
    semaphore_acquired = False
    if semaphore._value <= 0:
        raise HTTPException(
            status_code=429,
            detail="System is currently at maximum capacity for ingestion. Please try again later."
        )
    await semaphore.acquire()
    semaphore_acquired = True

    try:
        if not tenant_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please provide a valid API key via the X-API-Key header to upload documents.",
            )

        job_id, doc_id, source_path, canonical_url, content_hash, temp_dir, format_type, total_size = await prepare_ingestion(url, file, tenant_id, registry)

        if content_hash:
            existing_doc = registry.get_document_by_hash(tenant_id, content_hash)
            if existing_doc and existing_doc["status"] == "complete":
                logger.info(f"Skipping duplicate ingestion: hash {content_hash} already exists for tenant {tenant_id}")
                if temp_dir:
                    shutil.rmtree(temp_dir)
                dummy_job_id = str(uuid.uuid4())
                registry.register_job(dummy_job_id, existing_doc["doc_id"], existing_doc["source"], existing_doc["format"], tenant_id, content_hash)
                registry.update_job_status(dummy_job_id, "complete", 100)
                return {"job_id": dummy_job_id, "status": "complete"}

        old_doc = registry.get_document(doc_id)
        if old_doc:
            old_chunk_ids = registry.delete_document(doc_id)
            if old_chunk_ids:
                try:
                    if retriever:
                        vector_store = getattr(retriever, "vector_store", getattr(getattr(retriever, "dense_retriever", None), "vector_store", None))
                        if vector_store:
                            await asyncio.to_thread(vector_store.delete_chunks, chunk_ids=old_chunk_ids)
                except Exception as e:
                    logger.warning(f"Failed to delete old chunks from Qdrant: {e}")

        registry.register_job(job_id, doc_id, canonical_url if canonical_url else source_path, format_type, tenant_id, content_hash)

        if pipeline_logger:
            pipeline_logger.log_event("ingestion_started", job_id=job_id, tenant_id=tenant_id, source_type=format_type, source_ref=canonical_url if canonical_url else source_path)

        background_tasks.add_task(
            _process_ingestion,
            job_id, doc_id, source_path, tenant_id, extract_visuals, temp_dir,
            registry, semaphore, retriever, canonical_url, content_hash,
            getattr(request.app.state, "embedding_generator", None), pipeline_logger
        )

        response = {"job_id": job_id, "status": "queued"}
        if format_type == "sitemap":
            response["warning"] = "Sitemap detected. Pages will be crawled sequentially."
        elif file and total_size:
            estimated_chunks = total_size / 2000
            if estimated_chunks > 500:
                response["warning"] = f"Large document (~{int(estimated_chunks)} estimated chunks). Ingestion may take 8-10 minutes due to API rate limits. Partial results will be available progressively."

        semaphore_acquired = False
        return response

    except Exception:
        raise
    finally:
        if semaphore_acquired:
            semaphore.release()


@router.get("/ingest/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    registry: DocumentRegistry = Depends(get_registry)
):
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    metadata = job.get("metadata")
    if metadata and isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = None

    response = JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_pct=job["progress_pct"],
        error=job["error"],
        doc_id=job["doc_id"],
        metadata=metadata
    )

    if job["status"] in ("complete", "partial_success"):
        doc = registry.get_document(job["doc_id"])
        if doc:
            response.chunk_count = len(doc.get("chunk_ids", []))

    return response
