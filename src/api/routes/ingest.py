import os
import shutil
import tempfile
import logging
import traceback
from typing import Optional
import uuid
import asyncio
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form,
    HTTPException,
    BackgroundTasks,
    Request,
)
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.ingestion.dispatcher import IngestionDispatcher
from src.ingestion.pipeline import IncrementalIngestionPipeline
from src.api.startup import app_state, _init_components

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: int
    error: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_count: Optional[int] = None


async def _process_ingestion(
    job_id: str,
    doc_id: str,
    source_path: str,
    visibility: str,
    tenant_id: str,
    extract_visuals: bool,
    temp_dir: Optional[str],
    canonical_url: Optional[str] = None,
):
    try:
        semaphore = app_state.get("ingestion_semaphore")
        if not semaphore:
            semaphore = asyncio.Semaphore(3)  # Fallback if state is weird

        async with semaphore:
            registry = app_state.get("registry")
            dispatcher = IngestionDispatcher()

            # In the future, extract_visuals will be passed to dispatcher
            # For now we just pass it as kwargs if dispatcher expects it, else ignore
            adapter_result = await dispatcher.ingest(
                source_path, extract_visuals=extract_visuals
            )

        if canonical_url and adapter_result and adapter_result.documents:
            for doc in adapter_result.documents:
                doc.url = canonical_url

        if not adapter_result or not adapter_result.documents:
            if registry:
                registry.update_job_status(
                    job_id, "failed", error="Failed to extract content from source."
                )
            if temp_dir:
                shutil.rmtree(temp_dir)
            return

        pipeline = IncrementalIngestionPipeline()
        # pipeline.run_async will be implemented next, but for now we run sync in executor
        # We need to adapt pipeline.run to accept registry and job_id

        loop = asyncio.get_event_loop()

        # Get the pre-loaded embedding model to prevent event loop starvation and disk I/O on every ingestion
        retriever = app_state.get("retriever")
        embedding_model = None
        if retriever:
            if hasattr(retriever, "dense_retriever"):
                embedding_model = retriever.dense_retriever.model
            elif hasattr(retriever, "model"):
                embedding_model = retriever.model

        result = await loop.run_in_executor(
            None,
            pipeline.run,
            adapter_result.documents,
            visibility,
            tenant_id,
            registry,
            job_id,
            doc_id,
            adapter_result.visual_chunks,
            embedding_model,
        )

        if temp_dir:
            shutil.rmtree(temp_dir)

        pipeline_version = result.get("version")

        # We no longer re-initialize all components here, as ChromaDB refreshes internally
        # and reloading the cross-encoder and models from disk caused severe performance issues.

    except Exception as e:
        logger.error(f"Error during async ingestion: {e}\n{traceback.format_exc()}")
        registry = app_state.get("registry")
        if registry:
            registry.update_job_status(job_id, "failed", error=str(e))
        if temp_dir:
            shutil.rmtree(temp_dir)


@router.post("/ingest")
@limiter.limit("30/minute")
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    file: UploadFile = File(None),
    visibility: str = Form("private"),
    tenant_id: Optional[str] = Form(None),
    extract_visuals: bool = Form(False),
):
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide either url or file")

    if visibility == "private" and not tenant_id:
        raise HTTPException(
            status_code=400, detail="tenant_id is required for private visibility"
        )

    registry = app_state.get("registry")
    if not registry:
        raise HTTPException(
            status_code=503, detail="Service initializing, try again in 15s"
        )

    if tenant_id:
        quota = registry.get_tenant_quota(tenant_id)
        if quota >= 2000:
            raise HTTPException(
                status_code=429,
                detail="Tenant quota exceeded (2000 chunks max). Please delete old documents to ingest new ones.",
            )

    job_id = str(uuid.uuid4())

    # Generate deterministic doc_id based on URL or file name and tenant_id
    import hashlib

    source_ident = url if url else file.filename
    if tenant_id:
        source_ident = f"{tenant_id}_{source_ident}"
    doc_id = hashlib.md5(source_ident.encode()).hexdigest()

    temp_dir = None
    source_path = None
    canonical_url = None

    if url:
        source_path = url
        format_type = "web"
    elif file:
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, file.filename)
        with open(source_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        format_type = (
            file.filename.split(".")[-1].lower() if "." in file.filename else "unknown"
        )
        canonical_url = f"upload://{doc_id}/{file.filename}"

    # Deduplicate old chunks and jobs
    old_doc = registry.get_document(doc_id)
    if old_doc:
        old_chunk_ids = registry.delete_document(doc_id)
        if old_chunk_ids:
            try:
                retriever = app_state.get("retriever")
                if retriever:
                    vector_store = getattr(retriever, "vector_store", None)
                    if not vector_store and hasattr(retriever, "dense_retriever"):
                        vector_store = getattr(
                            retriever.dense_retriever, "vector_store", None
                        )
                    if vector_store:
                        vector_store.collection.delete(ids=old_chunk_ids)
            except Exception as e:
                logger.warning(f"Failed to delete old chunks from Chroma: {e}")

    registry.register_job(
        job_id,
        doc_id,
        canonical_url if canonical_url else source_path,
        format_type,
        visibility,
        tenant_id,
    )

    background_tasks.add_task(
        _process_ingestion,
        job_id,
        doc_id,
        source_path,
        visibility,
        tenant_id,
        extract_visuals,
        temp_dir,
        canonical_url,
    )

    return {"job_id": job_id, "status": "queued"}


@router.get("/ingest/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    registry = app_state.get("registry")
    if not registry:
        raise HTTPException(status_code=500, detail="Registry not initialized")

    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_pct=job["progress_pct"],
        error=job["error"],
        doc_id=job["doc_id"],
    )

    if job["status"] == "complete":
        doc = registry.get_document(job["doc_id"])
        if doc:
            response.chunk_count = len(doc.get("chunk_ids", []))

    return response


@router.post("/admin/rebuild-bm25")
async def rebuild_bm25(background_tasks: BackgroundTasks):
    from pathlib import Path
    from src.ingestion.pipeline import IncrementalIngestionPipeline

    if not Path("data/bm25_dirty.flag").exists():
        return {"status": "skipped", "message": "BM25 index is not marked dirty."}

    background_tasks.add_task(IncrementalIngestionPipeline()._rebuild_bm25)
    return {"status": "queued", "message": "BM25 rebuild queued."}
