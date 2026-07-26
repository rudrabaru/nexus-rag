import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

class JobStoreMixin:
    """Mixin handling job-related registry operations."""

    def _save_metadata(self, conn, job_id: str, new_meta: Optional[Dict[str, Any]]):
        if new_meta is None:
            return
        cursor = conn.execute("SELECT metadata FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        existing_meta = {}
        if row and row[0]:
            try:
                existing = json.loads(row[0])
                if isinstance(existing, dict):
                    existing_meta = existing
            except Exception:
                pass
        if isinstance(new_meta, dict):
            existing_meta.update(new_meta)
            final_str = json.dumps(existing_meta)
        else:
            final_str = json.dumps(new_meta)
        conn.execute("UPDATE jobs SET metadata = ? WHERE job_id = ?", (final_str, job_id))

    def register_job(
        self,
        job_id: str,
        doc_id: str,
        source: str,
        format: str,
        tenant_id: str,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
                status = CASE WHEN documents.status = 'complete' THEN 'complete' ELSE 'pending' END,
                updated_at = excluded.updated_at
            """,
                (doc_id, source, format, tenant_id, now, now, content_hash),
            )

            # Create job
            conn.execute(
                """
                INSERT INTO jobs (job_id, doc_id, status, progress_pct, created_at, metadata)
                VALUES (?, ?, 'queued', 0, ?, ?)
            """,
                (job_id, doc_id, now, json.dumps(metadata) if metadata else None),
            )

            conn.commit()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress_pct: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

            self._save_metadata(conn, job_id, metadata)

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
                UPDATE jobs SET status = ?, progress_pct = 100, finished_at = ?
                WHERE job_id = ?
            """,
                (status, now, job_id),
            )

            self._save_metadata(conn, job_id, metadata)

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
