"""
Clean, chunk and embed documents that are already Markdown (fetched pages or parsed uploads).

It writes nothing to storage: the caller commits the returned outcome in one transaction
(src/jobs/commit.py).
"""
import logging
import re
import time
from typing import Any, Callable, List, Optional

from src.chunking.chunker import DocumentChunker
from src.chunking.metadata import ChunkingConfig
from src.config import get_settings
from src.crawling.metadata import CrawledDocument
from src.embedding.generator import EmbeddingGenerator
from src.embedding.providers import build_embedder
from src.ingestion.embedding_worker import EmbeddingOutcome, EmbeddingWorker
from src.ingestion.errors import UnprocessableSourceError
from src.processing.cleaner import DocumentCleaner
from src.processing.models import Block

logger = logging.getLogger(__name__)

_LOG_LINE = re.compile(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}")
# Cleaning that leaves less than this many characters is treated as a cleaner failure, not as
# an empty document: the raw Markdown is chunked instead, so nothing is lost.
MIN_CLEANED_CHARS = 50


def _cleaned_markdown(cleaner: DocumentCleaner, doc: CrawledDocument, blocks: List[Block]) -> str:
    """The document without the blocks the cleaner removed; each removal is logged with its reason."""
    kept = cleaner.clean_document_blocks(blocks)
    cleaned = "\n\n".join(b.content for b in kept).strip()
    removed = [b for b in blocks if b.is_removed]
    logger.info(
        f"CLEAN | {doc.url} | words {len(doc.markdown_content.split())} -> {len(cleaned.split())} | "
        f"blocks removed {len(removed)}/{len(blocks)}"
    )
    for b in removed:
        logger.debug(f"CLEAN | removed [{b.removal_reason}] {b.content[:80]!r}")
    if len(cleaned) < MIN_CLEANED_CHARS:
        logger.warning(f"CLEAN | {doc.url} | cleaning left {len(cleaned)} chars; chunking the raw Markdown instead.")
        return doc.markdown_content
    return cleaned


def process_documents(
    crawled_docs: List[CrawledDocument],
    tenant_id: str,
    doc_id: str,
    on_progress: Optional[Callable[[int], None]] = None,
    pipeline_logger: Any = None,
    job_id: Optional[str] = None,
    embedding_generator: Optional[EmbeddingGenerator] = None,
) -> EmbeddingOutcome:
    """
    embedding_generator defaults to a fresh one per run: it keeps per-run state (last_error)
    that must not be shared between ingestions running concurrently in one worker.
    """
    if not tenant_id or not doc_id:
        raise ValueError("tenant_id and doc_id are required for ingestion")

    logger.info(f"Starting ingestion of {len(crawled_docs)} documents.")
    if sum(len(doc.markdown_content) for doc in crawled_docs) < MIN_CLEANED_CHARS:
        raise UnprocessableSourceError("Extracted content is too short or empty.")

    for doc in crawled_docs:
        lines = doc.markdown_content.splitlines()
        if lines and sum(1 for line in lines if _LOG_LINE.search(line)) / len(lines) > 0.4:
            logger.warning(f"Document {doc.url} appears to be a log file. Retrieval quality may be degraded.")

    def update_progress(pct: int):
        if on_progress:
            on_progress(pct)

    update_progress(55)
    start_time = time.time()

    # Block document frequency is only measurable across the documents of one job (a sitemap).
    cleaner = DocumentCleaner(total_documents=len(crawled_docs))
    all_blocks = [cleaner.parse_blocks(doc.markdown_content) for doc in crawled_docs]
    if len(crawled_docs) > 1:
        cleaner.process_corpus_frequencies(all_blocks)

    chunk_input_docs = [
        {"url": doc.url, "title": doc.title, "markdown_content": _cleaned_markdown(cleaner, doc, blocks)}
        for doc, blocks in zip(crawled_docs, all_blocks)
    ]

    update_progress(65)
    chunker = DocumentChunker(config=ChunkingConfig(source_version="v_live", output_version="v_live"))
    all_chunks = chunker.chunk_batch(chunk_input_docs)
    if not all_chunks:
        raise UnprocessableSourceError("No chunk could be produced from the extracted content.")
    for c in all_chunks:
        c.visibility = "private"
        c.tenant_id = tenant_id
        c.doc_id = doc_id

    update_progress(75)

    generator = embedding_generator or EmbeddingGenerator(build_embedder(get_settings()))
    outcome = EmbeddingWorker(generator).embed(all_chunks, update_progress, pipeline_logger, job_id)

    if pipeline_logger:
        pipeline_logger.log_event(
            "ingestion_audit",
            job_id=job_id,
            tenant_id=tenant_id,
            source=crawled_docs[0].url if crawled_docs else "unknown",
            index_id=generator.embedder.index_id,
            docs_crawled=len(crawled_docs),
            blocks_parsed=sum(len(blocks) for blocks in all_blocks),
            blocks_removed=sum(1 for blocks in all_blocks for b in blocks if b.is_removed),
            chunks_created=len(all_chunks),
            chunks_by_type={
                ct: sum(1 for c in all_chunks if c.content_type == ct) for ct in ["text", "code", "table", "mixed"]
            },
            chunks_embedded=len(outcome.chunks),
            total_tokens=outcome.total_tokens,
            pipeline_latency_seconds=round(time.time() - start_time, 2),
        )
    return outcome
