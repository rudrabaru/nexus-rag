import logging
import time
import re
from typing import List, Dict, Any

from src.crawling.metadata import CrawledDocument, VisualChunkDraft
from src.processing.cleaner import DocumentCleaner
from src.chunking.metadata import ChunkMetadata, ChunkingConfig
from src.processing.models import ProcessedDocument, Block
from src.processing.validator import ProcessingValidator
from src.chunking.chunker import DocumentChunker
from src.embedding.config import EmbeddingConfig
from src.embedding.generator import EmbeddingGenerator
from src.retrieving.vector_store import QdrantManager
from src.ingestion.embedding_worker import EmbeddingWorker

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
    def __init__(self, embedding_generator=None, db_manager=None):
        if db_manager is None:
            try:
                self.db_manager = QdrantManager()
                logger.info("Pipeline initialized new QdrantManager.")
            except Exception as e:
                logger.critical(f"QdrantManager failed to initialize in pipeline: {e}")
                raise e
        else:
            self.db_manager = db_manager
            
        self.embedding_generator = embedding_generator

    def run(
        self, crawled_docs: List[CrawledDocument], tenant_id: str, registry=None, job_id: str = None, doc_id: str = None, visual_chunks: List[VisualChunkDraft] = None, pipeline_logger: Any = None,
    ) -> Dict[str, Any]:
        if not tenant_id:
            raise ValueError("tenant_id is required for ingestion")

        logger.info(f"Starting incremental ingestion for {len(crawled_docs)} documents.")

        total_chars = sum(len(doc.markdown_content) for doc in crawled_docs)
        if total_chars < 50:
            raise ValueError("Extracted content is too short or empty. If this is a scanned PDF, vision extraction was also attempted by the adapter but returned no content. Ensure a GEMINI_API_KEY is configured, or upload a text-based PDF.")

        _LOG_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}")
        for doc in crawled_docs:
            lines = doc.markdown_content.splitlines()
            if lines:
                log_lines = sum(1 for line in lines if _LOG_PATTERN.search(line))
                if log_lines / len(lines) > 0.4:
                    logger.warning(f"Document {doc.url} appears to be a log file. Retrieval quality may be degraded.")

        def update_progress(pct: int, status="processing"):
            if registry and job_id:
                registry.update_job_status(job_id, status, pct)

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
                    visibility="private", tenant_id=tenant_id,
                )
                all_chunks.append(v_meta)

        update_progress(75)

        generator = self.embedding_generator or EmbeddingGenerator(EmbeddingConfig())
        worker = EmbeddingWorker(generator, self.db_manager)
        
        result = worker.process_batches(all_chunks, tenant_id, registry, job_id, pipeline_logger)
        total_added = result["total_added"]

        duration = time.time() - start_time

        if job_id:
            EmbeddingWorker.write_audit_log(job_id, crawled_docs, all_blocks, all_chunks, result["total_tokens"], duration)

        return {
            "status": "success",
            "version": "v_live",
            "docs_processed": len(crawled_docs),
            "chunks_added": total_added,
            "latency_seconds": round(duration, 2),
        }
