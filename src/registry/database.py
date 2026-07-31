import sqlite3
import logging
from pathlib import Path
import threading
from contextlib import contextmanager

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
            default_path = "/data/registry.db" if os.environ.get("SPACE_ID") else "data/registry.db"
            db_path = os.environ.get("REGISTRY_DB_PATH", default_path)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._local = threading.local()
        
        self._init_db()

    def _get_thread_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=15.0)
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode per connection to allow concurrent readers
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    @contextmanager
    def _get_conn(self):
        conn = self._get_thread_conn()
        with conn:
            yield conn

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
                conn.execute("ALTER TABLE jobs ADD COLUMN metadata TEXT")
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

            try:
                conn.execute("ALTER TABLE api_keys ADD COLUMN total_embedding_tokens INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

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

    def rebuild_registry_from_qdrant(self, qdrant_manager):
        """
        Reconstructs the SQLite registry from Qdrant payloads.
        Useful for Render Free Tier where SQLite is wiped on restart but Qdrant persists.
        """
        import datetime
        import json
        from collections import defaultdict
        
        offset = None
        total_points = 0
        doc_chunks = defaultdict(list)
        doc_tenants = {}
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        with self._get_conn() as conn:
            # Clear existing FTS to avoid duplicates on partial wipe
            conn.execute("DELETE FROM fts_chunks")
            
            while True:
                response, offset = qdrant_manager.client.scroll(
                    collection_name=qdrant_manager.collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                
                if not response:
                    break
                    
                total_points += len(response)
                
                for point in response:
                    payload = point.payload or {}
                    tenant_id = payload.get("tenant_id", "unknown")
                    chunk_id = payload.get("chunk_id")
                    source_document = payload.get("source_document")
                    chunk_text = payload.get("document_text", "")
                    
                    if not chunk_id or not source_document:
                        continue
                        
                    doc_chunks[source_document].append(chunk_id)
                    doc_tenants[source_document] = tenant_id
                    
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO fts_chunks (chunk_id, tenant_id, source_document, chunk_text) VALUES (?, ?, ?, ?)",
                            (chunk_id, tenant_id, source_document, chunk_text)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to insert chunk {chunk_id} into fts: {e}")
                
                # Commit the batch of FTS points
                conn.commit()
                
                if offset is None:
                    break
                    
            # Now insert the document records based on the aggregated chunks
            for source_document, chunk_ids in doc_chunks.items():
                tenant_id = doc_tenants.get(source_document, "unknown")
                conn.execute(
                    """
                    INSERT OR IGNORE INTO documents (
                        doc_id, source, format, status, visibility, 
                        tenant_id, ingested_at, updated_at, chunk_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_document, source_document, "reconstructed", "complete", "private",
                        tenant_id, now, now, json.dumps(chunk_ids)
                    )
                )
            conn.commit()
            
        return total_points
