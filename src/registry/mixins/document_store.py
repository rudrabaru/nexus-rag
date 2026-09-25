from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select, update

from src.registry.rows import row_to_dict
from src.registry.schema import chunks, documents


def _chunk_count():
    return (
        select(func.count())
        .where(chunks.c.doc_id == documents.c.doc_id)
        .correlate(documents)
        .scalar_subquery()
        .label("chunk_count")
    )


def _documents_with_counts():
    return select(documents, _chunk_count())


class DocumentStoreMixin:
    """
    Document rows. A document's chunk count is derived from the chunks table rather than
    stored beside it, so the two cannot disagree.
    """

    def _fetch_one(self, stmt) -> Optional[Dict[str, Any]]:
        with self._engine.connect() as conn:
            return row_to_dict(conn.execute(stmt).first())

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._fetch_one(_documents_with_counts().where(documents.c.doc_id == doc_id))

    def get_document_by_hash(self, tenant_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        if not content_hash:
            return None
        return self._fetch_one(
            _documents_with_counts().where(
                documents.c.tenant_id == tenant_id, documents.c.content_hash == content_hash
            )
        )

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists one tenant's documents, or every document when tenant_id is None (admin only)."""
        stmt = _documents_with_counts().order_by(documents.c.ingested_at)
        if tenant_id:
            stmt = stmt.where(documents.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            return [row_to_dict(row) for row in conn.execute(stmt)]

    def get_tenant_quota(self, tenant_id: str) -> int:
        """Chunks currently stored for the tenant."""
        stmt = select(func.count()).select_from(chunks).where(chunks.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    def get_doc_count(self, tenant_id: str) -> int:
        stmt = select(func.count()).select_from(documents).where(documents.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    def set_content_hash(self, doc_id: str, content_hash: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(update(documents).where(documents.c.doc_id == doc_id).values(content_hash=content_hash))

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document; its jobs, chunks, vectors and sparse entries cascade in the same transaction."""
        with self._engine.begin() as conn:
            return conn.execute(delete(documents).where(documents.c.doc_id == doc_id)).rowcount > 0
