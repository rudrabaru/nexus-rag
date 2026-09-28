"""
The ingest task: parse, chunk, embed, commit. Runs on the heavy worker (Dockerfile.worker).

This worker never contacts a website. A URL ingestion's pages were fetched by the fetch
worker (src/jobs/fetch_tasks.py) and wait in fetched_pages; an upload's bytes wait in
ingest_sources. Everything this module imports is worker-only; the API only knows the task's
name, queue and payload (src/jobs/contract.py).

Failure handling:
- UnprocessableSourceError means retrying would produce the same result (no usable content,
  content too large). It is not re-raised: the job is marked failed and Procrastinate sees
  the task as having run successfully, so it is never retried.
- Any other exception is re-raised so Procrastinate retries it with backoff. On the attempt
  that exhausts the retry budget, the job is also marked failed here, so the domain status
  (jobs.status) does not stay "processing" once Procrastinate has given up.
"""
import asyncio
import hashlib
import logging
import os
import tempfile
from typing import List

import procrastinate

from src.crawling.metadata import CrawledDocument
from src.ingestion.errors import UnprocessableSourceError
from src.ingestion.pipeline import process_documents
from src.jobs.commit import commit_ingestion
from src.jobs.contract import INGEST_QUEUE, INGEST_TASK, INGEST_TASK_NAME, MAX_RETRIES, RECOVERY_CRON, RECOVERY_TASK_NAME, IngestionRequest
from src.jobs.recovery import recover_stalled_jobs
from src.jobs.support import Progress, already_finished, job_tag, pipeline_logger
from src.parsing.files import parse_file
from src.registry.database import DocumentRegistry
from src.registry.engine import get_sync_engine

logger = logging.getLogger(__name__)

blueprint = procrastinate.Blueprint()

MAX_CHARS_PER_INGESTION = 1_500_000  # protects the embedding API from a single runaway document


def _fetched_documents(request: IngestionRequest, registry: DocumentRegistry) -> List[CrawledDocument]:
    pages = registry.get_fetched_pages(request.job_id)
    if not pages:
        raise UnprocessableSourceError(f"No fetched pages found for job {request.job_id} (already processed or discarded).")
    return [CrawledDocument(url=p["url"], title=p["title"] or p["url"], markdown_content=p["markdown"]) for p in pages]


def _uploaded_document(request: IngestionRequest, registry: DocumentRegistry, progress: Progress) -> List[CrawledDocument]:
    source = registry.get_ingest_source(request.job_id)
    if source is None:
        raise UnprocessableSourceError(f"No uploaded content found for job {request.job_id} (already processed or expired).")
    filename, content = source
    with tempfile.TemporaryDirectory(prefix="nexus-ingest-") as temp_dir:
        path = os.path.join(temp_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        parsed = parse_file(path)
    progress.set(50, {"parser": parsed.parser})
    logger.info(f"{job_tag(request)} parsed {filename!r} with {parsed.parser} | chars={len(parsed.markdown)}")
    return [CrawledDocument(url=request.source_ref, title=filename, markdown_content=parsed.markdown)]


async def _ingest(request: IngestionRequest) -> None:
    tag = job_tag(request)
    sync_engine = get_sync_engine()
    registry = DocumentRegistry(sync_engine)
    progress = Progress(registry, request.job_id)

    logger.info(f"{tag} STAGE 1 | load | source={request.source_ref!r}")
    progress.set(5)
    if request.url:
        documents = await asyncio.to_thread(_fetched_documents, request, registry)
    else:
        documents = await asyncio.to_thread(_uploaded_document, request, registry, progress)

    total_chars = sum(len(d.markdown_content) for d in documents)
    logger.info(f"{tag} STAGE 1 DONE | documents={len(documents)} total_chars={total_chars}")
    if total_chars > MAX_CHARS_PER_INGESTION:
        raise UnprocessableSourceError(
            f"Document is too large. Extracted text ({total_chars} characters) exceeds the {MAX_CHARS_PER_INGESTION} limit."
        )

    # Duplicate detection for URL sources (uploads are hashed and checked by the API).
    if not request.content_hash:
        content_hash = hashlib.sha256("".join(d.markdown_content for d in documents).encode()).hexdigest()
        existing = await asyncio.to_thread(registry.get_document_by_hash, request.tenant_id, content_hash)
        if existing and existing["status"] == "complete" and existing["doc_id"] != request.doc_id:
            logger.info(f"{tag} same content as document {existing['doc_id'][:8]}; completing as a duplicate.")
            await asyncio.to_thread(registry.complete_as_duplicate, request.job_id, existing["doc_id"])
            return
        await asyncio.to_thread(registry.set_content_hash, request.doc_id, content_hash)

    logger.info(f"{tag} STAGE 2 | pipeline | docs={len(documents)} chars={total_chars}")
    outcome = await asyncio.to_thread(
        process_documents,
        documents, request.tenant_id, request.doc_id, progress.set, pipeline_logger(), request.job_id,
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
    registry = DocumentRegistry(get_sync_engine())

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
    report = await recover_stalled_jobs(context.app.job_manager, DocumentRegistry(get_sync_engine()), INGEST_QUEUE, INGEST_TASK)
    if report.requeued or report.failed:
        logger.warning(f"Stalled-job sweep: requeued={report.requeued} failed={report.failed}")
