import logging
import time
import re
from typing import Any, Callable, List, Optional

from src.crawling.metadata import CrawledDocument, VisualChunkDraft
from src.processing.cleaner import DocumentCleaner
from src.chunking.metadata import ChunkMetadata, ChunkingConfig
from src.processing.models import ProcessedDocument, Block
from src.processing.validator import ProcessingValidator
from src.chunking.chunker import DocumentChunker
from src.embedding.config import EmbeddingConfig
from src.embedding.generator import EmbeddingGenerator
from src.ingestion.embedding_worker import EmbeddingOutcome, EmbeddingWorker
from src.ingestion.errors import UnprocessableSourceError

logger = logging.getLogger(__name__)

def _strip_intra_document_repeats(blocks: List[Block]) -> List[Block]:
    from collections import Counter
    content_counts = Counter(
        b.content_hash for b in blocks 
        if not b.metrics.is_code and not b.metrics.is_table and 3 <= b.metrics.word_count < 30
    )
    repeats = {h for h, count in content_counts.items() if count >= 3}
    cleaned_blocks = []
    for b in blocks:
        if b.content_hash in repeats and 3 <= b.metrics.word_count < 30:
            b.is_removed = True
            b.removal_reason = "Intra-document repeat (PDF header/footer)"
        cleaned_blocks.append(b)
    return cleaned_blocks

class IncrementalIngestionPipeline:
    """
    Parses, cleans, chunks and embeds documents. It writes nothing to storage: the caller
    commits the returned outcome in one transaction (src/jobs/commit.py).
    """

    def __init__(self, embedding_generator=None):
        # A generator per pipeline run by default: it keeps per-run state (stats, last_error)
        # that must not be shared between ingestions running concurrently in one worker.
        self.embedding_generator = embedding_generator

    def run(
        self,
        crawled_docs: List[CrawledDocument],
        tenant_id: str,
        doc_id: str,
        visual_chunks: Optional[List[VisualChunkDraft]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
        pipeline_logger: Any = None,
        job_id: Optional[str] = None,
    ) -> EmbeddingOutcome:
        if not tenant_id or not doc_id:
            raise ValueError("tenant_id and doc_id are required for ingestion")

        logger.info(f"Starting incremental ingestion for {len(crawled_docs)} documents.")

        total_chars = sum(len(doc.markdown_content) for doc in crawled_docs)
        if total_chars < 50:
            raise UnprocessableSourceError("Extracted content is too short or empty. If this is a scanned PDF, vision extraction was also attempted by the adapter but returned no content. Ensure a GEMINI_API_KEY is configured, or upload a text-based PDF.")

        _LOG_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}")
        for doc in crawled_docs:
            lines = doc.markdown_content.splitlines()
            if lines:
                log_lines = sum(1 for line in lines if _LOG_PATTERN.search(line))
                if log_lines / len(lines) > 0.4:
                    logger.warning(f"Document {doc.url} appears to be a log file. Retrieval quality may be degraded.")

        def update_progress(pct: int):
            if on_progress:
                on_progress(pct)

        update_progress(55)
        start_time = time.time()

        cleaner = DocumentCleaner(total_documents=len(crawled_docs))
        all_blocks = []
        for doc in crawled_docs:
            blocks = cleaner.parse_blocks(doc.markdown_content)
            if doc.url.lower().endswith(".pdf"):
                blocks = _strip_intra_document_repeats(blocks)
            all_blocks.append(blocks)

        if len(crawled_docs) > 1:
            cleaner.process_corpus_frequencies(all_blocks)

        processed_docs = []
        for i, doc in enumerate(crawled_docs):
            blocks = all_blocks[i]
            cleaned_blocks = cleaner.clean_document_blocks(blocks)
            pdoc = ProcessedDocument(**doc.model_dump())
            pdoc.blocks = blocks
            pdoc.page_category = "incremental_doc"
            ProcessingValidator.validate_document(doc.markdown_content, cleaned_blocks, pdoc)
            processed_docs.append(pdoc)

        update_progress(65)
        chunker = DocumentChunker(config=ChunkingConfig(source_version="v_live", output_version="v_live"))
        chunk_input_docs = []
        for pdoc in processed_docs:
            doc_dict = pdoc.model_dump()
            cleaned = pdoc.cleaned_markdown.strip()
            if len(cleaned) >= 50:
                doc_dict["markdown_content"] = cleaned
            else:
                logger.warning(f"cleaned_markdown for '{pdoc.url}' is too short; falling back to raw markdown_content.")
            chunk_input_docs.append(doc_dict)

        all_chunks = chunker.chunk_batch(chunk_input_docs)
        for c in all_chunks:
            c.visibility = "private"
            c.tenant_id = tenant_id
            c.doc_id = doc_id
            prefix = f"[{c.source_document} > {' > '.join(c.heading_path or [])}]\n"
            c.embedding_text = prefix + c.chunk_text

        if visual_chunks and crawled_docs:
            parent_doc = crawled_docs[0]
            start_index = len(all_chunks)
            for i, vc in enumerate(visual_chunks):
                est_tokens = int(len(vc.text.split()) * 1.3)
                v_meta = ChunkMetadata(
                    chunk_id=f"{parent_doc.url}_visual_{i}", source_url=parent_doc.url, source_document=parent_doc.title or parent_doc.url,
                    title=parent_doc.title or "Unknown", chunk_index=start_index + i, total_chunks=start_index + len(visual_chunks),
                    chunk_text=vc.text, token_count=est_tokens, char_start=0, char_end=0, content_type="visual_description",
                    visual_asset_ref=vc.asset_ref, visual_asset_type=vc.asset_type, document_version="v_live", chunk_version="v_live",
                    visibility="private", tenant_id=tenant_id, doc_id=doc_id,
                )
                all_chunks.append(v_meta)

        update_progress(75)

        generator = self.embedding_generator or EmbeddingGenerator(EmbeddingConfig())
        outcome = EmbeddingWorker(generator).embed(all_chunks, update_progress, pipeline_logger, job_id)

        if pipeline_logger:
            pipeline_logger.log_event(
                "ingestion_audit",
                job_id=job_id,
                tenant_id=tenant_id,
                source=crawled_docs[0].url if crawled_docs else "unknown",
                docs_crawled=len(crawled_docs),
                blocks_parsed=sum(len(blocks) for blocks in all_blocks),
                blocks_removed=sum(1 for doc_blocks in all_blocks for b in doc_blocks if getattr(b, "is_removed", False)),
                chunks_created=len(all_chunks),
                chunks_by_type={
                    ct: sum(1 for c in all_chunks if getattr(c, "content_type", "") == ct)
                    for ct in ["text", "code", "table", "mixed", "visual_description"]
                },
                chunks_embedded=len(outcome.chunks),
                total_tokens=outcome.total_tokens,
                pipeline_latency_seconds=round(time.time() - start_time, 2),
            )
        return outcome
