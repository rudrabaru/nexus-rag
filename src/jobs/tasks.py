"""
The ingestion task: fetch, parse, chunk, embed, commit.

Worker-only. This module (and everything it imports — the dispatcher, the pipeline, the
embedding generator) never loads in the API process; the API only knows the task's name,
queue and payload shape (src/jobs/contract.py).

Failure handling:
- UnprocessableSourceError means retrying would produce the same result (no usable content,
  content too large). It is not re-raised: the job is marked failed and Procrastinate sees
  the task as having run successfully, so it is never retried.
- Any other exception is re-raised so Procrastinate's RetryStrategy (src/jobs/queue.py)
  retries it with backoff. On the attempt that exhausts the retry budget, the job is also
  marked failed here, so the domain status (jobs.status) does not stay "processing" once
  Procrastinate has given up.
"""
import asyncio
import hashlib
import logging
import tempfile

import procrastinate

from src.ingestion.dispatcher import IngestionDispatcher
from src.ingestion.errors import UnprocessableSourceError
from src.ingestion.pipeline import IncrementalIngestionPipeline
from src.ingestion.sitemap_fetch import MAX_SITEMAP_PAGES, fetch_sitemap_pages
from src.jobs.commit import commit_ingestion
from src.jobs.contract import (
    INGEST_QUEUE,
    INGEST_TASK_NAME,
    MAX_RETRIES,
    RECOVERY_CRON,
    RECOVERY_TASK_NAME,
    IngestionRequest,
)
from src.jobs.recovery import recover_stalled_jobs
from src.jobs.support import Progress, already_finished, pipeline_logger, resolve_source
from src.registry.database import DocumentRegistry
from src.registry.engine import get_sync_engine
from src.retrieving.chunk_writes import existing_source_urls

logger = logging.getLogger(__name__)

blueprint = procrastinate.Blueprint()

MAX_CHARS_PER_INGESTION = 1_500_000  # protects the LLM/embedding APIs from a single runaway document


async def _ingest(request: IngestionRequest) -> None:
    tag = f"[job={request.job_id[:8]} doc={request.doc_id[:8]}]"
    sync_engine = get_sync_engine()
    registry = DocumentRegistry(sync_engine)
    progress = Progress(registry, request.job_id)
    dispatcher = IngestionDispatcher()

    with tempfile.TemporaryDirectory(prefix="nexus-ingest-") as temp_dir:
        source_path = await asyncio.to_thread(resolve_source, request, registry, temp_dir)

        logger.info(f"{tag} STAGE 1 | dispatch | source={source_path!r} extract_visuals={request.extract_visuals}")
        adapter_result = await dispatcher.ingest(source_path, extract_visuals=request.extract_visuals)
        doc_count = len(adapter_result.documents) if adapter_result else 0
        total_chars = sum(len(d.markdown_content) for d in (adapter_result.documents if adapter_result else []))
        logger.info(f"{tag} STAGE 1 DONE | documents={doc_count} total_chars={total_chars}")

        is_empty_sitemap_shell = bool(
            adapter_result and adapter_result.documents and all(d.markdown_content == "" for d in adapter_result.documents)
        )
        if not is_empty_sitemap_shell:
            progress.set(50)

        # ── Sitemap: the dispatcher returns one empty CrawledDocument per URL; fetch each ──
        if is_empty_sitemap_shell:
            urls = [d.url for d in adapter_result.documents]
            if len(urls) > MAX_SITEMAP_PAGES:
                logger.warning(f"{tag} Sitemap has {len(urls)} URLs; truncating to {MAX_SITEMAP_PAGES}.")
                urls = urls[:MAX_SITEMAP_PAGES]

            existing_urls = set()
            if request.resume:
                try:
                    with sync_engine.connect() as conn:
                        existing_urls = existing_source_urls(conn, request.tenant_id, request.doc_id)
                    if existing_urls:
                        logger.info(f"{tag} Resuming: {len(existing_urls)} pages already indexed.")
                except Exception as e:
                    logger.warning(f"{tag} Could not fetch existing URLs for resume: {e}")

            progress.set(5, {"total_pages": len(urls), "indexed_pages": 0, "failed_pages": 0})
            docs, visual_chunks, failed_reasons = await fetch_sitemap_pages(
                dispatcher, urls, existing_urls, request.extract_visuals, progress.set, tag
            )
            if failed_reasons:
                progress.set(50, {"error_reason": f"{len(failed_reasons)}/{len(urls)} pages failed: " + "; ".join(failed_reasons[:2])})
            adapter_result.documents = docs
            adapter_result.visual_chunks = visual_chunks
            total_chars = sum(len(d.markdown_content) for d in docs)
            logger.info(f"{tag} Sitemap fetch done | fetched={len(docs)} total_chars={total_chars}")

        # ── Uploaded files keep their canonical upload:// URL, not the adapter's local path ──
        if not request.url and adapter_result and adapter_result.documents:
            canonical = request.source_ref
            for doc in adapter_result.documents:
                doc.url = canonical

        if not adapter_result or not adapter_result.documents:
            raise UnprocessableSourceError("Failed to extract content from source.")
        if total_chars > MAX_CHARS_PER_INGESTION:
            raise UnprocessableSourceError(
                f"Document is too large. Extracted text ({total_chars} characters) exceeds the {MAX_CHARS_PER_INGESTION} limit."
            )

        # ── Duplicate detection for URL sources (uploads are hashed and checked by the API) ──
        content_hash = request.content_hash
        if not content_hash:
            combined = "".join(d.markdown_content for d in adapter_result.documents)
            content_hash = hashlib.sha256(combined.encode()).hexdigest()
            existing = await asyncio.to_thread(registry.get_document_by_hash, request.tenant_id, content_hash)
            if existing and existing["status"] == "complete":
                logger.info(f"{tag} Duplicate content detected — marking complete without re-storing.")
                await asyncio.to_thread(registry.update_job_status, request.job_id, "complete", 100)
                return
            await asyncio.to_thread(registry.set_content_hash, request.doc_id, content_hash)

        logger.info(f"{tag} STAGE 2 | pipeline | docs={len(adapter_result.documents)} chars={total_chars}")
        progress.set(55)
        pipeline = IncrementalIngestionPipeline()
        outcome = await asyncio.to_thread(
            pipeline.run,
            adapter_result.documents,
            request.tenant_id,
            request.doc_id,
            adapter_result.visual_chunks,
            progress.set,
            pipeline_logger(),
            request.job_id,
        )

        await asyncio.to_thread(commit_ingestion, sync_engine, request.job_id, request.tenant_id, outcome)
        logger.info(f"{tag} DONE | status={outcome.status} chunks={len(outcome.chunks)} tokens={outcome.total_tokens}")


@blueprint.task(
    name=INGEST_TASK_NAME,
    queue=INGEST_QUEUE,
    retry=procrastinate.RetryStrategy(max_attempts=MAX_RETRIES, exponential_wait=5),
    pass_context=True,
)
def ingest_document(context: procrastinate.JobContext, **kwargs) -> None:
    request = IngestionRequest(**kwargs)
    sync_engine = get_sync_engine()
    registry = DocumentRegistry(sync_engine)

    skip_reason = already_finished(registry, request.job_id)
    if skip_reason:
        logger.info(f"[job={request.job_id[:8]}] not running: {skip_reason}.")
        return

    try:
        asyncio.run(_ingest(request))
    except UnprocessableSourceError as e:
        logger.error(f"[job={request.job_id[:8]}] unprocessable source, not retrying: {e}")
        registry.fail_job(request.job_id, str(e))
    except Exception as e:
        is_last_attempt = context.job.attempts >= MAX_RETRIES
        logger.error(
            f"[job={request.job_id[:8]}] attempt {context.job.attempts + 1}/{MAX_RETRIES + 1} failed: {e}",
            exc_info=True,
        )
        if is_last_attempt:
            registry.fail_job(request.job_id, f"Ingestion failed after {context.job.attempts + 1} attempts: {e}")
        raise


@blueprint.periodic(cron=RECOVERY_CRON, periodic_id="recover-stalled-ingestions")
@blueprint.task(name=RECOVERY_TASK_NAME, queue=INGEST_QUEUE, lock=RECOVERY_TASK_NAME, pass_context=True)
async def recover_stalled_ingestions(context: procrastinate.JobContext, timestamp: int) -> None:
    """Requeues (or fails, once out of retries) ingestion jobs whose worker died. lock= keeps two workers' sweeps from overlapping."""
    report = await recover_stalled_jobs(context.app.job_manager, DocumentRegistry(get_sync_engine()))
    if report.requeued or report.failed:
        logger.warning(f"Stalled-job sweep: requeued={report.requeued} failed={report.failed}")
