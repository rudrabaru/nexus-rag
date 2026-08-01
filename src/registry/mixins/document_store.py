import json
from typing import List, Dict, Any, Optional

class DocumentStoreMixin:
    """Mixin handling document-related registry operations."""

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None

            doc = dict(row)
            doc["chunk_ids"] = json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
            doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
            return doc

    def get_document_by_hash(self, tenant_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        if not content_hash:
            return None
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE tenant_id = ? AND content_hash = ?", (tenant_id, content_hash))
            row = cursor.fetchone()
            if not row:
                return None
            doc = dict(row)
            doc["chunk_ids"] = json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
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
            doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
            return doc


    def get_tenant_quota(self, tenant_id: str) -> int:
        query = "SELECT SUM(json_array_length(chunk_ids)) FROM documents WHERE tenant_id = ?"
        with self._get_conn() as conn:
            cursor = conn.execute(query, [tenant_id])
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def list_documents(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if tenant_id:
            query = "SELECT * FROM documents WHERE tenant_id = ?"
            params = [tenant_id]
        else:
            query = "SELECT * FROM documents"
            params = []

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            docs = []
            for row in cursor.fetchall():
                doc = dict(row)
                doc["chunk_ids"] = (
                    json.loads(doc["chunk_ids"]) if doc["chunk_ids"] else []
                )
                doc["stats"] = json.loads(doc["stats"]) if doc["stats"] else {}
                docs.append(doc)
            return docs

    def delete_document(self, doc_id: str) -> List[str]:
        """Deletes a document and returns its chunk_ids so the caller can remove them from Qdrant."""
        doc = self.get_document(doc_id)
        if not doc:
            return []

        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM jobs WHERE doc_id = ?", (doc_id,))
            if doc["chunk_ids"]:
                placeholders = ",".join("?" for _ in doc["chunk_ids"])
                conn.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})", doc["chunk_ids"])
            conn.commit()

        return doc["chunk_ids"]

    def get_doc_count(self, tenant_id: str) -> int:
        """Returns the total number of documents for a given tenant."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE tenant_id = ?", (tenant_id,))
            row = cursor.fetchone()
            return row[0] if row else 0
