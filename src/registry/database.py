import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """
    SQLite-backed Document Registry for tracking documents and ingestion jobs.
    Handles concurrent access safely via connection pooling/isolation in sqlite3.
    """

    def __init__(self, db_path: str = ".chroma_db/registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT,
                    format TEXT,
                    status TEXT,
                    visibility TEXT,
                    tenant_id TEXT,
                    ingested_at TEXT,
                    updated_at TEXT,
                    chunk_ids TEXT,
                    asset_ids TEXT,
                    stats TEXT,
                    error TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    status TEXT,
                    progress_pct INTEGER,
                    created_at TEXT,
                    finished_at TEXT,
                    error TEXT
                )
            """)
            conn.commit()

    def register_job(
        self,
        job_id: str,
        doc_id: str,
        source: str,
        format: str,
        visibility: str,
        tenant_id: Optional[str] = None,
    ):
        """Registers a new job and creates/updates a pending document entry."""
        now = datetime.utcnow().isoformat()
        with self._get_conn() as conn:
            # Upsert document as pending
            conn.execute(
                """
                INSERT INTO documents (doc_id, source, format, status, visibility, tenant_id, ingested_at, updated_at, chunk_ids, asset_ids, stats)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, '[]', '[]', '{}')
                ON CONFLICT(doc_id) DO UPDATE SET
                status = 'pending',
                updated_at = excluded.updated_at,
                visibility = excluded.visibility
            """,
                (doc_id, source, format, visibility, tenant_id, now, now),
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
            datetime.utcnow().isoformat() if status in ["complete", "failed"] else None
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
                    (error, datetime.utcnow().isoformat(), job_id),
                )

            conn.commit()

    def reset_stuck_jobs(self):
        """Finds any jobs left in 'queued' or 'processing' states (e.g. from a server restart) and marks them as failed."""
        now = datetime.utcnow().isoformat()
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
    ):
        """Marks a job as complete and updates the document with final metadata."""
        now = datetime.utcnow().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = 'complete', progress_pct = 100, finished_at = ?
                WHERE job_id = ?
            """,
                (now, job_id),
            )

            conn.execute(
                """
                UPDATE documents SET 
                    status = 'complete',
                    updated_at = ?,
                    chunk_ids = ?,
                    asset_ids = ?,
                    stats = ?
                WHERE doc_id = (SELECT doc_id FROM jobs WHERE job_id = ?)
            """,
                (
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

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None

            doc = dict(row)
            doc["chunk_ids"] = json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
            doc["asset_ids"] = json.loads(doc["asset_ids"]) if doc["asset_ids"] else []
            doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
            return doc

    def get_document_by_source_and_tenant(
        self, source: str, tenant_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM documents WHERE source = ?"
        params = [source]
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        else:
            query += ' AND (tenant_id IS NULL OR tenant_id = "")'

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None

            doc = dict(row)
            doc["chunk_ids"] = json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
            doc["asset_ids"] = json.loads(doc["asset_ids"]) if doc["asset_ids"] else []
            doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
            return doc

    def get_document_by_source_and_visibility(
        self, source: str, visibility: str
    ) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM documents WHERE source = ? AND visibility = ?"
        with self._get_conn() as conn:
            cursor = conn.execute(query, [source, visibility])
            row = cursor.fetchone()
            if not row:
                return None

            doc = dict(row)
            doc["chunk_ids"] = json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
            doc["asset_ids"] = json.loads(doc["asset_ids"]) if doc["asset_ids"] else []
            doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
            return doc

    def get_tenant_quota(self, tenant_id: str) -> int:
        query = "SELECT chunk_ids FROM documents WHERE tenant_id = ?"
        total_chunks = 0
        with self._get_conn() as conn:
            cursor = conn.execute(query, [tenant_id])
            for row in cursor.fetchall():
                chunk_ids = json.loads(row["chunk_ids"]) if row["chunk_ids"] else []
                total_chunks += len(chunk_ids)
        return total_chunks

    def list_documents(
        self, tenant_id: Optional[str] = None, visibility: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM documents WHERE 1=1"
        params = []
        if tenant_id:
            query += ' AND (tenant_id = ? OR visibility = "public")'
            params.append(tenant_id)
        if visibility:
            query += " AND visibility = ?"
            params.append(visibility)

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            docs = []
            for row in cursor.fetchall():
                doc = dict(row)
                doc["chunk_ids"] = (
                    json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
                )
                doc["asset_ids"] = (
                    json.loads(doc["asset_ids"]) if doc["asset_ids"] else []
                )
                doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
                docs.append(doc)
            return docs

    def delete_document(self, doc_id: str) -> List[str]:
        """Deletes a document and returns its chunk_ids so the caller can remove them from ChromaDB."""
        doc = self.get_document(doc_id)
        if not doc:
            return []

        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM jobs WHERE doc_id = ?", (doc_id,))
            conn.commit()

        return doc["chunk_ids"]
