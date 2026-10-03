from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import Engine

from src.db.rows import row_to_dict
from src.db.schema import chunks, documents


def _documents_with_counts():
    """Document rows with their chunk count, derived from the chunks table so the two cannot disagree."""
    chunk_count = (
        select(func.count()).where(chunks.c.doc_id == documents.c.doc_id).correlate(documents).scalar_subquery()
    )
    return select(documents, chunk_count.label("chunk_count"))


class DocumentStore:
    def __init__(self, engine: Engine):
        self._engine = engine

    def _fetch_one(self, stmt) -> Optional[Dict[str, Any]]:
        with self._engine.connect() as conn:
            return row_to_dict(conn.execute(stmt).first())

    def _list(self, *conditions) -> List[Dict[str, Any]]:
        stmt = _documents_with_counts().where(*conditions).order_by(documents.c.ingested_at)
        with self._engine.connect() as conn:
            return [row_to_dict(row) for row in conn.execute(stmt)]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._fetch_one(_documents_with_counts().where(documents.c.doc_id == doc_id))

    def get_document_by_hash(self, tenant_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        if not content_hash:
            return None
        return self._fetch_one(
            _documents_with_counts().where(documents.c.tenant_id == tenant_id, documents.c.content_hash == content_hash)
        )

    def list_documents(self, tenant_id: str) -> List[Dict[str, Any]]:
        """One tenant's documents. A missing tenant is an error, never "all tenants"."""
        if not tenant_id:
            raise ValueError("list_documents needs a tenant_id; use list_all_documents for an admin listing.")
        return self._list(documents.c.tenant_id == tenant_id)

    def list_all_documents(self) -> List[Dict[str, Any]]:
        """Every tenant's documents. For the admin only."""
        return self._list()

    def chunk_count(self, tenant_id: str) -> int:
        """Chunks currently stored for the tenant (the storage quota's measure)."""
        stmt = select(func.count()).select_from(chunks).where(chunks.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    def document_count(self, tenant_id: str) -> int:
        stmt = select(func.count()).select_from(documents).where(documents.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    def set_content_hash(self, doc_id: str, content_hash: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(update(documents).where(documents.c.doc_id == doc_id).values(content_hash=content_hash))

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document; its jobs, chunks, vectors and keyword entries cascade in the same transaction."""
        with self._engine.begin() as conn:
            return conn.execute(delete(documents).where(documents.c.doc_id == doc_id)).rowcount > 0
