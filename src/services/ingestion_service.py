import asyncio
import os
import shutil
import logging
import traceback
import hashlib
import time
import uuid
import tempfile
from typing import Optional, Any, Tuple
from fastapi import UploadFile, HTTPException

from src.ingestion.dispatcher import IngestionDispatcher
from src.ingestion.pipeline import IncrementalIngestionPipeline
from src.registry.database import DocumentRegistry

logger = logging.getLogger(__name__)


async def prepare_ingestion(
    url: Optional[str],
    file: Optional[UploadFile],
    tenant_id: str,
    registry: DocumentRegistry
) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str], str]:
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide either url or file")

    quota = registry.get_tenant_quota(tenant_id)
    if quota >= 2000:
        raise HTTPException(
            status_code=429,
            detail="Tenant quota exceeded (2000 chunks max). Please delete old documents to ingest new ones.",
        )

    job_id = str(uuid.uuid4())
    source_ident = url if url else file.filename
    source_ident = f"{tenant_id}_{source_ident}"
    doc_id = hashlib.blake2b(source_ident.encode(), digest_size=16).hexdigest()

    temp_dir = None
    source_path = None
    canonical_url = None
    content_hash = None
    total_size = 0

    if url:
        source_path = url
        if url.lower().endswith(".xml") or "sitemap" in url.lower():
            format_type = "sitemap"
        else:
            format_type = "web"
    elif file:
        safe_filename = os.path.basename(file.filename)
        allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
        ext = os.path.splitext(safe_filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed types are: {', '.join(allowed_extensions)}")

        MAX_UPLOAD_BYTES = 20 * 1024 * 1024
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, safe_filename)
        hasher = hashlib.sha256()
        
        with open(source_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_BYTES:
                    buffer.close()
                    shutil.rmtree(temp_dir)
                    raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_BYTES / (1024*1024):.0f}MB limit")
                hasher.update(chunk)
                buffer.write(chunk)
                
        content_hash = hasher.hexdigest()
        format_type = safe_filename.split(".")[-1].lower() if "." in safe_filename else "unknown"
        canonical_url = f"upload://{doc_id}/{safe_filename}"

    return job_id, doc_id, source_path, canonical_url, content_hash, temp_dir, format_type, total_size

async def _process_ingestion(
    job_id: str,
    doc_id: str,
    source_path: str,
    tenant_id: str,
    extract_visuals: bool,
    temp_dir: Optional[str],
    registry: DocumentRegistry,
    semaphore: asyncio.Semaphore,
    retriever: Any,
    canonical_url: Optional[str] = None,
    content_hash: Optional[str] = None,
    embedding_generator: Any = None,
    pipeline_logger: Any = None,
    resume: bool = False,
):
    start_time = time.time()
    tag = f"[job={job_id[:8]} doc={doc_id[:8]}]"

    try:
        # ── STAGE 1: Adapter dispatch ─────────────────────────────────────────
        logger.info(
            f"{tag} STAGE 1 | Adapter dispatch | source={source_path!r} "
            f"extract_visuals={extract_visuals} canonical_url={canonical_url!r}"
        )
        dispatcher = IngestionDispatcher()

        adapter_result = await dispatcher.ingest(
            source_path, extract_visuals=extract_visuals, registry=registry, job_id=job_id, doc_id=doc_id
        )

        doc_count = len(adapter_result.documents) if adapter_result else 0
        total_chars = sum(len(d.markdown_content) for d in (adapter_result.documents if adapter_result else []))
        logger.info(
            f"{tag} STAGE 1 DONE | documents={doc_count} total_chars={total_chars} "
            f"visual_chunks={len(adapter_result.visual_chunks) if adapter_result else 0}"
        )

        if registry and not (adapter_result and adapter_result.documents and all(d.markdown_content == "" for d in adapter_result.documents)):
            registry.update_job_status(job_id, "processing", 50)

        # ── STAGE 1.5: Sitemap bounded concurrent fetch ────────────────────────
        if adapter_result and adapter_result.documents and all(d.markdown_content == "" for d in adapter_result.documents) and len(adapter_result.documents) > 0:
            urls = [doc.url for doc in adapter_result.documents]
            total = len(urls)
            logger.info(f"{tag} STAGE 1.5 | Sitemap bounded concurrent fetch | total_pages={total}")
            if registry:
                registry.update_job_status(job_id, "processing", 5, metadata={"total_pages": total, "indexed_pages": 0, "failed_pages": 0})
            
            all_docs = []
            all_visual_chunks = []
            failed_pages = 0
            processed = 0
            failed_reasons = []
            sem = asyncio.Semaphore(4)

            existing_urls = set()
            if resume and retriever:
                try:
                    vector_store = getattr(retriever, "vector_store", getattr(getattr(retriever, "dense_retriever", None), "vector_store", None))
                    if vector_store and hasattr(vector_store, "get_existing_urls"):
                        # Get existing URLs sequentially to avoid breaking the current event loop context
                        # Since qdrant client scroll is sync under the hood, we can wrap it or run to thread.
                        # Wait, get_existing_urls is synchronous.
                        existing_urls = await asyncio.to_thread(vector_store.get_existing_urls, tenant_id, doc_id)
                        if existing_urls:
                            logger.info(f"{tag} Resuming ingestion: Found {len(existing_urls)} previously indexed pages.")
                except Exception as e:
                    logger.warning(f"{tag} Failed to fetch existing URLs for resume: {e}")

            async def fetch_url(u):
                nonlocal processed, failed_pages
                if u in existing_urls:
                    processed += 1
                    logger.info(f"{tag} Skipping previously indexed URL: {u}")
                    if registry:
                        pct = 5 + int(45 * processed / total)
                        registry.update_job_status(job_id, "processing", pct, metadata={"total_pages": total, "indexed_pages": processed - failed_pages, "failed_pages": failed_pages})
                    return None, None

                async with sem:
                    try:
                        res = await dispatcher.web_adapter.ingest(u, extract_visuals=extract_visuals)
                        processed += 1
                        if registry:
                            pct = 5 + int(45 * processed / total)
                            registry.update_job_status(job_id, "processing", pct, metadata={"total_pages": total, "indexed_pages": processed - failed_pages, "failed_pages": failed_pages})
                        return res, None
                    except Exception as e:
                        logger.warning(f"{tag} Sitemap skipped URL {u}: {e}")
                        failed_pages += 1
                        processed += 1
                        if registry:
                            pct = 5 + int(45 * processed / total)
                            registry.update_job_status(job_id, "processing", pct, metadata={"total_pages": total, "indexed_pages": processed - failed_pages, "failed_pages": failed_pages})
                        return None, f"{u}: {str(e)[:60]}"

            results = await asyncio.gather(*(fetch_url(u) for u in urls))
            for res, err in results:
                if res and res.documents:
                    all_docs.extend(res.documents)
                if res and res.visual_chunks:
                    all_visual_chunks.extend(res.visual_chunks)
                if err:
                    failed_reasons.append(err)
            
            if failed_reasons and registry:
                err_summary = f"{failed_pages}/{total} pages failed scraping: " + "; ".join(failed_reasons[:2])
                registry.update_job_status(job_id, "processing", 50, metadata={"error_reason": err_summary})

            adapter_result.documents = all_docs
            adapter_result.visual_chunks = all_visual_chunks
            doc_count = len(adapter_result.documents)
            total_chars = sum(len(d.markdown_content) for d in adapter_result.documents)
            logger.info(f"{tag} STAGE 1.5 DONE | successfully fetched documents={doc_count} total_chars={total_chars} failed_pages={failed_pages}")

        # ── STAGE 2: URL canonicalisation ─────────────────────────────────────
        is_sitemap = source_path.lower().endswith(".xml") or "sitemap" in source_path.lower()
        if canonical_url and adapter_result and adapter_result.documents and not is_sitemap:
            for doc in adapter_result.documents:
                old_url = doc.url
                doc.url = canonical_url
                logger.debug(f"{tag} STAGE 2 | URL rewrite: {old_url!r} → {canonical_url!r}")

        # ── GATE: Empty adapter result ─────────────────────────────────────────
        if not adapter_result or not adapter_result.documents:
            logger.error(
                f"{tag} GATE FAIL | Adapter returned no documents. "
                f"source={source_path!r} extract_visuals={extract_visuals}. "
                "Check STAGE 1 logs for the root cause."
            )
            if registry:
                registry.update_job_status(
                    job_id, "failed", error="Failed to extract content from source."
                )
            if temp_dir:
                shutil.rmtree(temp_dir)
            return

        # ── GATE: Empty content after adapter ─────────────────────────────────
        if total_chars < 50:
            logger.error(
                f"{tag} GATE FAIL | Adapter returned documents but total content "
                f"is only {total_chars} chars. For scanned PDFs the vision fallback "
                "should have recovered content — check STAGE 1 logs for vision errors."
            )

        # ── STAGE 3: Duplicate detection ──────────────────────────────────────
        if not content_hash:
            combined_text = "".join(doc.markdown_content for doc in adapter_result.documents)
            content_hash = hashlib.sha256(combined_text.encode()).hexdigest()
            logger.info(f"{tag} STAGE 3 | Content hash computed: {content_hash[:16]}...")
            if registry:
                existing_doc = registry.get_document_by_hash(tenant_id, content_hash)
                if existing_doc and existing_doc["status"] == "complete":
                    logger.info(f"{tag} STAGE 3 | Duplicate detected — skipping ingestion.")
                    registry.update_job_status(job_id, "complete", 100)
                    if temp_dir:
                        shutil.rmtree(temp_dir)
                    return
                with registry._get_conn() as conn:
                    conn.execute("UPDATE documents SET content_hash = ? WHERE doc_id = ?", (content_hash, doc_id))

        # ── STAGE 4: Pipeline (chunk → embed → store) ─────────────────────────
        db_manager = getattr(retriever, "vector_store", getattr(getattr(retriever, "dense_retriever", None), "vector_store", None))
        pipeline = IncrementalIngestionPipeline(embedding_generator=embedding_generator, db_manager=db_manager)
        loop = asyncio.get_running_loop()

        logger.info(
            f"{tag} STAGE 4 | Starting pipeline: "
            f"docs={len(adapter_result.documents)} chars={total_chars}"
        )

        result = await loop.run_in_executor(
            None,
            pipeline.run,
            adapter_result.documents,
            tenant_id,
            registry,
            job_id,
            doc_id,
            adapter_result.visual_chunks,
            pipeline_logger,
        )

        if temp_dir:
            shutil.rmtree(temp_dir)

        chunks_stored = result.get("chunks_added", result.get("chunks_stored", "?"))
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"{tag} STAGE 4 DONE | chunks_stored={chunks_stored} "
            f"total_duration_ms={duration_ms:.0f}"
        )

        if pipeline_logger:
            doc = registry.get_document(doc_id)
            chunk_count = len(doc.get("chunk_ids", [])) if doc else 0
            pipeline_logger.log_event(
                "ingestion_complete",
                job_id=job_id,
                chunk_count=chunk_count,
                duration_ms=duration_ms
            )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"{tag} INGESTION FAILED after {duration_ms:.0f}ms | "
            f"error={e!r}\n{traceback.format_exc()}"
        )
        if registry:
            registry.update_job_status(job_id, "failed", error=str(e))
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        semaphore.release()
