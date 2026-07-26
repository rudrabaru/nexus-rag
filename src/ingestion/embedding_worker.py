import time
import logging
from typing import List, Any
import os
import json

logger = logging.getLogger(__name__)

class EmbeddingWorker:
    """Handles the batching and processing of chunks for embeddings and database insertion."""

    def __init__(self, embedding_generator, db_manager):
        self.embedding_generator = embedding_generator
        self.db_manager = db_manager

    def process_batches(self, all_chunks: List[Any], tenant_id: str, registry: Any = None, job_id: str = None, pipeline_logger: Any = None) -> dict:
        embed_start_time = time.time()
        
        BATCH_SIZE = 50
        total_added = 0
        all_failed_indices = []
        embedded_chunk_ids = []
        total_tokens = 0
        total_embedded_chunks = 0
        
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            embedded_batch, failed_indices = self.embedding_generator.generate_embeddings(batch)
            
            if failed_indices:
                all_failed_indices.extend([idx + i for idx in failed_indices])
                
            if embedded_batch:
                added = self.db_manager.load_chunks(embedded_batch)
                total_added += added
                if registry:
                    registry.insert_fts_chunks([chunk.model_dump() for chunk in embedded_batch])
                
                embedded_chunk_ids.extend([c.chunk_id for c in embedded_batch])
                total_tokens += sum(c.token_count for c in embedded_batch)
                total_embedded_chunks += len(embedded_batch)
                
        embed_duration_ms = (time.time() - embed_start_time) * 1000
        if pipeline_logger:
            pipeline_logger.log_event(
                "embedding_complete",
                job_id=job_id,
                chunk_count=total_embedded_chunks,
                batch_count=(len(all_chunks) // BATCH_SIZE) + (1 if len(all_chunks) % BATCH_SIZE else 0),
                duration_ms=embed_duration_ms
            )

        logger.info(f"Upserted {total_added} chunks to Qdrant.")

        if registry and job_id:
            job_status = "partial_success" if all_failed_indices else "complete"
            metadata = None
            err_reason = None
            if all_failed_indices:
                err_reason = getattr(self.embedding_generator, "last_error", None) or f"{len(all_failed_indices)} chunks failed embedding (API rate limit or timeout)."
                metadata = {
                    "failed_chunk_indices": all_failed_indices,
                    "embedded": len(all_chunks) - len(all_failed_indices),
                    "total": len(all_chunks),
                    "error_reason": err_reason,
                }

            registry.complete_job(
                job_id,
                embedded_chunk_ids,
                [],
                {"total_added": total_added, "total_tokens": total_tokens},
                status=job_status,
                metadata=metadata,
                error=err_reason,
            )
            registry.increment_tenant_embedding_tokens(tenant_id, total_tokens)
            
        return {
            "total_added": total_added,
            "total_tokens": total_tokens,
            "job_status": job_status if job_id else "complete",
            "failed_indices": all_failed_indices
        }

    @staticmethod
    def write_audit_log(job_id, crawled_docs, all_blocks, all_chunks, total_tokens, duration):
        from pathlib import Path
        
        audit = {
            "job_id": job_id,
            "source": crawled_docs[0].url if crawled_docs else "unknown",
            "docs_crawled": len(crawled_docs),
            "blocks_parsed": sum(len(blocks) for blocks in all_blocks),
            "blocks_removed": sum(
                1 for doc_blocks in all_blocks for b in doc_blocks if getattr(b, 'is_removed', False)
            ),
            "chunks_created": len(all_chunks),
            "chunks_by_type": {
                ct: sum(1 for c in all_chunks if getattr(c, 'content_type', '') == ct)
                for ct in ["text", "code", "table", "mixed"]
            },
            "total_tokens": total_tokens,
            "pipeline_latency_seconds": round(duration, 2),
        }
        base_dir = os.environ.get("OBSERVABILITY_DIR")
        if not base_dir:
            if os.environ.get("REGISTRY_DB_PATH", "").startswith("/data"):
                base_dir = "/data/observability"
            else:
                base_dir = "observability"
        obs_path = Path(base_dir) / f"ingestion_{job_id[:8]}.json"
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_path.write_text(json.dumps(audit, indent=2))
        logger.info(f"Ingestion audit written to {obs_path}")
