"""
The ingest task: parse, chunk, embed, commit. Runs on the heavy worker (Dockerfile.worker).

This worker never contacts a website. A URL ingestion's pages were fetched by the fetch
worker (src/jobs/fetch_tasks.py) and wait in fetched_pages; an upload's bytes wait in
ingest_sources. Everything this module imports is worker-only; the API only knows the task's
name, queue and payload (src/jobs/contract.py).

Failures and the recovery of dead workers follow the shared policy in src/jobs/policy.py.
"""
import asyncio
import hashlib
import logging
import os
import re
import tempfile
from typing import List

import procrastinate

from src.crawling.metadata import CrawledDocument
from src.errors import UnprocessableSourceError
from src.ingestion.pipeline import process_documents
from src.jobs.commit import commit_ingestion
from src.jobs.contract import INGEST_QUEUE, INGEST_TASK, INGEST_TASK_NAME, MAX_RETRIES, RECOVERY_TASK_NAME, IngestionRequest
from src.jobs.policy import register_recovery, run_with_policy
from src.jobs.support import Progress, job_tag, pipeline_logger
from src.parsing.files import parse_file
from src.db.engine import get_sync_engine
from src.stores.documents import DocumentStore
from src.stores.fetches import FetchStore
from src.stores.jobs import JobStore

logger = logging.getLogger(__name__)

blueprint = procrastinate.Blueprint()

MAX_CHARS_PER_INGESTION = 1_500_000  # protects the embedding API from a single runaway document


def _fetched_documents(request: IngestionRequest, fetches: FetchStore) -> List[CrawledDocument]:
    pages = fetches.get_fetched_pages(request.job_id)
    if not pages:
        raise UnprocessableSourceError(f"No fetched pages found for job {request.job_id} (already processed or discarded).")
    return [CrawledDocument(url=p["url"], title=p["title"] or p["url"], markdown_content=p["markdown"]) for p in pages]


def _uploaded_document(request: IngestionRequest, jobs: JobStore, progress: Progress) -> List[CrawledDocument]:
    source = jobs.get_ingest_source(request.job_id)
    if source is None:
        raise UnprocessableSourceError(f"No uploaded content found for job {request.job_id} (already processed or expired).")
    filename, content = source
    # The uploaded name is a label, never a path: this worker may run on another OS than the API,
    # where a name like "..\..\x.md" or "C:\x.docx" would escape the temp directory.
    suffix = os.path.splitext(re.split(r"[\\/]", filename)[-1])[1].lower()
    with tempfile.TemporaryDirectory(prefix="nexus-ingest-") as temp_dir:
        path = os.path.join(temp_dir, f"upload{suffix}")
        with open(path, "wb") as f:
            f.write(content)
        parsed = parse_file(path)
    progress.set(50, {"parser": parsed.parser})
    logger.info(f"{job_tag(request)} parsed {filename!r} with {parsed.parser} | chars={len(parsed.markdown)}")
    return [CrawledDocument(url=request.source_ref, title=filename, markdown_content=parsed.markdown)]


def _ingest(request: IngestionRequest) -> None:
    tag = job_tag(request)
    engine = get_sync_engine()
    documents_store, jobs, fetches = DocumentStore(engine), JobStore(engine), FetchStore(engine)
    progress = Progress(jobs, request.job_id)

    logger.info(f"{tag} STAGE 1 | load | source={request.source_ref!r}")
    progress.set(5)
    documents = _fetched_documents(request, fetches) if request.url else _uploaded_document(request, jobs, progress)

    total_chars = sum(len(d.markdown_content) for d in documents)
    logger.info(f"{tag} STAGE 1 DONE | documents={len(documents)} total_chars={total_chars}")
    if total_chars > MAX_CHARS_PER_INGESTION:
        raise UnprocessableSourceError(
            f"Document is too large. Extracted text ({total_chars} characters) exceeds the {MAX_CHARS_PER_INGESTION} limit."
        )

    # Duplicate detection for URL sources (uploads are hashed and checked by the API).
    if not request.content_hash:
        content_hash = hashlib.sha256("".join(d.markdown_content for d in documents).encode()).hexdigest()
        existing = documents_store.get_document_by_hash(request.tenant_id, content_hash)
        if existing and existing["status"] == "complete" and existing["doc_id"] != request.doc_id:
            logger.info(f"{tag} same content as document {existing['doc_id'][:8]}; completing as a duplicate.")
            jobs.complete_as_duplicate(request.job_id, existing["doc_id"])
            return
        documents_store.set_content_hash(request.doc_id, content_hash)

    logger.info(f"{tag} STAGE 2 | pipeline | docs={len(documents)} chars={total_chars}")
    outcome = process_documents(
        documents, request.tenant_id, request.doc_id, progress.set, pipeline_logger(), request.job_id,
    )

    commit_ingestion(engine, request, outcome)
    logger.info(f"{tag} DONE | status={outcome.status} chunks={len(outcome.chunks)} tokens={outcome.total_tokens}")


@blueprint.task(
    name=INGEST_TASK_NAME,
    queue=INGEST_QUEUE,
    retry=procrastinate.RetryStrategy(max_attempts=MAX_RETRIES, exponential_wait=5),
    pass_context=True,
)
async def ingest_document(context: procrastinate.JobContext, **kwargs) -> None:
    """The work blocks, so it runs in a thread: the event loop keeps sending the worker's heartbeat meanwhile."""
    request = IngestionRequest(**kwargs)
    await run_with_policy(context, request, lambda: asyncio.to_thread(_ingest, request), "ingestion")


register_recovery(blueprint, INGEST_QUEUE, INGEST_TASK, RECOVERY_TASK_NAME)
