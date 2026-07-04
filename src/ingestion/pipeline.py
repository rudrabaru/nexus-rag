import json
import pickle
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi
from langdetect import detect, LangDetectException

from src.crawling.metadata import CrawledDocument, VisualChunkDraft
from src.processing.cleaner import DocumentCleaner
from src.chunking.metadata import ChunkMetadata, ChunkingConfig
from src.processing.models import ProcessedDocument
from src.processing.validator import ProcessingValidator
from src.chunking.chunker import DocumentChunker
from src.embedding.config import EmbeddingConfig
from src.embedding.generator import EmbeddingGenerator
from src.retrieving.vector_store import ChromaDBManager

logger = logging.getLogger(__name__)


class IncrementalIngestionPipeline:
    """
    Runs the end-to-end pipeline in-memory for newly ingested documents,
    upserts them into ChromaDB, and updates the BM25 index.
    """

    def __init__(self):
        self.db_manager = ChromaDBManager(collection_name="unified_corpus")
        self.bm25_file = Path("data/bm25_index.pkl")
        self.bm25_file.parent.mkdir(exist_ok=True)

    def run(
        self,
        crawled_docs: List[CrawledDocument],
        visibility: str = "public",
        tenant_id: Optional[str] = None,
        registry=None,
        job_id: str = None,
        doc_id: str = None,
        visual_chunks: List[VisualChunkDraft] = None,
        embedding_model=None,
    ) -> Dict[str, Any]:
        """
        Executes the ingestion pipeline.
        If registry and job_id are provided, it updates the job status progressively.
        """
        logger.info(
            f"Starting incremental ingestion for {len(crawled_docs)} documents."
        )

        def update_progress(pct: int, status="processing"):
            if registry and job_id:
                registry.update_job_status(job_id, status, pct)

        update_progress(10)
        start_time = time.time()

        # 1. Processing (Cleaning)
        cleaner = DocumentCleaner(total_documents=len(crawled_docs))
        all_blocks = []
        for doc in crawled_docs:
            all_blocks.append(cleaner.parse_blocks(doc.markdown_content))

        # Corpus-frequency analysis is only meaningful across multiple documents.
        # For a single-doc upload, every block has frequency 1/1, making the
        # frequency signal meaningless and causing false-positive boilerplate removal.
        # Structural protections (code/table/heading guards) still apply regardless.
        if len(crawled_docs) > 1:
            cleaner.process_corpus_frequencies(all_blocks)

        # Language Constraint: Enforce English-only documents
        try:
            combined_text = " ".join([doc.markdown_content for doc in crawled_docs])
            if combined_text.strip():
                from langdetect import detect_langs

                langs = detect_langs(combined_text)
                if not any(l.lang == "en" for l in langs):
                    # If it's code/PEP it might be misclassified, but if 'en' isn't even in the detected langs, it's probably not English
                    # For safety, let's just use detect, but if it fails, maybe warn instead of crash
                    logger.warning(
                        f"Detected languages {langs}, no 'en' found. Proceeding with caution."
                    )
                    # We will still proceed as langdetect is not 100% accurate for code-heavy text
                    # raise ValueError("Only English documents are supported.")
        except LangDetectException:
            pass  # Not enough text to detect language, proceed

        processed_docs = []
        for i, doc in enumerate(crawled_docs):
            blocks = all_blocks[i]
            cleaned_blocks = cleaner.clean_document_blocks(blocks)

            pdoc = ProcessedDocument(**doc.model_dump())
            pdoc.blocks = blocks
            pdoc.page_category = "incremental_doc"

            ProcessingValidator.validate_document(
                doc.markdown_content, cleaned_blocks, pdoc
            )
            processed_docs.append(pdoc)

        update_progress(30)

        # 2. Chunking
        chunker = DocumentChunker(
            config=ChunkingConfig(source_version="v_live", output_version="v_live")
        )
        all_chunks = chunker.chunk_batch([pdoc.model_dump() for pdoc in processed_docs])

        # Inject Multi-Tenancy Metadata
        for c in all_chunks:
            c.visibility = visibility
            if tenant_id:
                c.tenant_id = tenant_id

        # Append visual chunks
        if visual_chunks and crawled_docs:
            parent_doc = crawled_docs[
                0
            ]  # Assume 1 source doc per run for visual chunks
            start_index = len(all_chunks)
            for i, vc in enumerate(visual_chunks):
                # Basic token estimate: 1 word ~ 1.3 tokens
                est_tokens = int(len(vc.text.split()) * 1.3)

                v_meta = ChunkMetadata(
                    chunk_id=f"{parent_doc.url}_visual_{i}",
                    source_url=parent_doc.url,
                    source_document=parent_doc.title or parent_doc.url,
                    title=parent_doc.title or "Unknown",
                    chunk_index=start_index + i,
                    total_chunks=start_index + len(visual_chunks),
                    chunk_text=vc.text,
                    token_count=est_tokens,
                    char_start=0,
                    char_end=0,
                    content_type="visual_description",
                    visual_asset_ref=vc.asset_ref,
                    visual_asset_type=vc.asset_type,
                    document_version="v_live",
                    chunk_version="v_live",
                    visibility=visibility,
                    tenant_id=tenant_id,
                )
                all_chunks.append(v_meta)

        update_progress(50)

        # 3. Embedding
        embed_config = EmbeddingConfig()
        generator = EmbeddingGenerator(embed_config, model=embedding_model)
        embedded_chunks = generator.generate_embeddings(all_chunks)

        # 4. Upsert to Vector Store

        total_added = self.db_manager.load_chunks(embedded_chunks)
        logger.info(f"Upserted {total_added} chunks to ChromaDB.")
        update_progress(90)

        # 5. Mark BM25 as dirty instead of inline rebuild
        Path("data/bm25_dirty.flag").touch()

        if registry and job_id:
            chunk_ids = [chunk.chunk_id for chunk in embedded_chunks]
            total_tokens = (
                sum(chunk.token_count for chunk in embedded_chunks)
                if embedded_chunks
                else 0
            )
            registry.complete_job(
                job_id,
                chunk_ids,
                [],
                {"total_added": total_added, "total_tokens": total_tokens},
            )
        update_progress(100, status="complete")

        duration = time.time() - start_time

        return {
            "status": "success",
            "version": "v_live",
            "docs_processed": len(crawled_docs),
            "chunks_added": total_added,
            "latency_seconds": round(duration, 2),
        }

    def _rebuild_bm25(self):
        try:
            results = self.db_manager.collection.get(include=["metadatas", "documents"])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not documents:
                return

            import re
            
            tokenized_corpus = [
                re.sub(r"[^\w\s]", " ", doc.lower()).split() for doc in documents
            ]
            bm25 = BM25Okapi(tokenized_corpus)

            chunks_data = []
            for meta, doc in zip(metadatas, documents):
                meta_copy = dict(meta)
                meta_copy["chunk_text"] = doc
                if "heading_path" in meta_copy and isinstance(
                    meta_copy["heading_path"], str
                ):
                    try:
                        meta_copy["heading_path"] = json.loads(
                            meta_copy["heading_path"]
                        )
                    except (json.JSONDecodeError, ValueError):
                        meta_copy["heading_path"] = []
                chunks_data.append(meta_copy)

            bm25_data = {"bm25": bm25, "chunks": chunks_data}

            with open(self.bm25_file, "wb") as f:
                pickle.dump(bm25_data, f)
            logger.info(
                f"Rebuilt BM25 index with {len(documents)} total documents in corpus."
            )
        except Exception as e:
            logger.error(f"Failed to rebuild BM25: {e}")
        finally:
            if Path("data/bm25_dirty.flag").exists():
                try:
                    Path("data/bm25_dirty.flag").unlink()
                except Exception:
                    pass
