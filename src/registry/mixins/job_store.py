import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

class JobStoreMixin:
    """Mixin handling job-related registry operations."""

    def register_job(
        self,
        job_id: str,
        doc_id: str,
        source: str,
        format: str,
        tenant_id: str,
        content_hash: Optional[str] = None,
    ):
        """Registers a new job and creates/updates a pending document entry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            # Upsert document as pending
            conn.execute(
                """
                INSERT INTO documents (doc_id, source, format, status, visibility, tenant_id, ingested_at, updated_at, chunk_ids, asset_ids, stats, content_hash)
                VALUES (?, ?, ?, 'pending', 'private', ?, ?, ?, '[]', '[]', '{}', ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                status = 'pending',
                updated_at = excluded.updated_at
            """,
                (doc_id, source, format, tenant_id, now, now, content_hash),
            )

            # Create job
            conn.execute(
                """
                INSERT INTO jobs (job_id, doc_id, status, progress_pct, created_at)
                VALUES (?, ?, 'queued', 0, ?)
            """,
                (job_id, doc_id, now),
            )

            conn.commit()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress_pct: int = 0,
        error: Optional[str] = None,
    ):
        """Updates the progress of a job."""
        now = (
            datetime.now(timezone.utc).isoformat() if status in ["complete", "failed"] else None
        )

        with self._get_conn() as conn:
            if status in ["complete", "failed"]:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, progress_pct = ?, finished_at = ?, error = ?
                    WHERE job_id = ?
                """,
                    (status, progress_pct, now, error, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, progress_pct = ?
                    WHERE job_id = ?
                """,
                    (status, progress_pct, job_id),
                )

            if status == "failed":
                conn.execute(
                    """
                    UPDATE documents SET status = 'failed', error = ?, updated_at = ?
                    WHERE doc_id = (SELECT doc_id FROM jobs WHERE job_id = ?)
                """,
                    (error, datetime.now(timezone.utc).isoformat(), job_id),
                )

            conn.commit()

    def reset_stuck_jobs(self):
        """Finds any jobs left in 'queued' or 'processing' states (e.g. from a server restart) and marks them as failed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            # Update documents first
            conn.execute(
                """
                UPDATE documents SET status = 'failed', error = 'Server restarted during ingestion', updated_at = ?
                WHERE doc_id IN (SELECT doc_id FROM jobs WHERE status IN ('queued', 'processing'))
            """,
                (now,),
            )

            # Update jobs
            conn.execute(
                """
                UPDATE jobs SET status = 'failed', error = 'Server restarted during ingestion', finished_at = ?
                WHERE status IN ('queued', 'processing')
            """,
                (now,),
            )

            conn.commit()

    def complete_job(
        self,
        job_id: str,
        chunk_ids: List[str],
        asset_ids: List[str],
        stats: Dict[str, Any],
        status: str = "complete",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Marks a job as complete (or partial_success) and updates the document with final metadata."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, progress_pct = 100, finished_at = ?, metadata = ?
                WHERE job_id = ?
            """,
                (status, now, json.dumps(metadata) if metadata else None, job_id),
            )

            conn.execute(
                """
                UPDATE documents SET 
                    status = ?,
                    updated_at = ?,
                    chunk_ids = ?,
                    asset_ids = ?,
                    stats = ?
                WHERE doc_id = (SELECT doc_id FROM jobs WHERE job_id = ?)
            """,
                (
                    status,
                    now,
                    json.dumps(chunk_ids),
                    json.dumps(asset_ids),
                    json.dumps(stats),
                    job_id,
                ),
            )

            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
