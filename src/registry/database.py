import sqlite3
import logging
from pathlib import Path
import threading
from contextlib import contextmanager
from typing import Optional

from .mixins.document_store import DocumentStoreMixin
from .mixins.job_store import JobStoreMixin
from .mixins.sparse_index import SparseIndexMixin

logger = logging.getLogger(__name__)

class DocumentRegistry(DocumentStoreMixin, JobStoreMixin, SparseIndexMixin):
    """
    SQLite-backed Document Registry for tracking documents and ingestion jobs.
    Handles concurrent access safely via connection pooling/isolation in sqlite3.
    Modularized via mixins for maintainability.
    """

    def __init__(self, db_path: str = None):
        import os
        if db_path is None:
            default_path = "/data/.chroma_db/registry.db" if os.environ.get("SPACE_ID") else ".chroma_db/registry.db"
            db_path = os.environ.get("REGISTRY_DB_PATH", default_path)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        
        self._init_db()

    @contextmanager
    def _get_conn(self):
        with self._lock:
            with self._conn:
                yield self._conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
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
                    error TEXT,
                    content_hash TEXT
                )
            """)
            
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    status TEXT,
                    progress_pct INTEGER,
                    created_at TEXT,
                    finished_at TEXT,
                    error TEXT,
                    metadata TEXT
                )
            """)
            
            try:
                conn.execute("ALTER TABLE api_keys ADD COLUMN total_embedding_tokens INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    created_at TEXT,
                    total_embedding_tokens INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    chunk_id,
                    tenant_id UNINDEXED,
                    source_document UNINDEXED,
                    chunk_text,
                    tokenize='porter'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS observability_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT,
                    timestamp TEXT,
                    query TEXT,
                    latency_ms REAL,
                    tokens_used INTEGER,
                    faithfulness_score REAL,
                    details TEXT
                )
            """)

            cost_columns = [
                ("embedding_tokens", "INTEGER"),
                ("embedding_cost_usd", "REAL"),
                ("generation_input_tokens", "INTEGER"),
                ("generation_output_tokens", "INTEGER"),
                ("generation_cost_usd", "REAL"),
                ("rerank_cost_usd", "REAL"),
                ("total_cost_usd", "REAL"),
            ]
            for col, col_type in cost_columns:
                try:
                    conn.execute(f"ALTER TABLE observability_logs ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT,
                    timestamp TEXT,
                    tenant_id TEXT,
                    query_id TEXT,
                    job_id TEXT,
                    details TEXT
                )
            """)

            conn.commit()

    def increment_tenant_embedding_tokens(self, tenant_id: str, tokens: int):
        if not tokens:
            return
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE api_keys SET total_embedding_tokens = total_embedding_tokens + ?
                WHERE tenant_id = ?
                """,
                (tokens, tenant_id)
            )
            conn.commit()
