import re
from typing import List, Dict, Any

class SparseIndexMixin:
    """Mixin handling FTS5 sparse indexing operations."""

    def insert_fts_chunks(self, chunks: List[Dict[str, Any]]):
        """Inserts processed chunks into the FTS5 virtual table."""
        if not chunks:
            return
        with self._get_conn() as conn:
            for chunk in chunks:
                conn.execute("DELETE FROM fts_chunks WHERE chunk_id = ?", (chunk["chunk_id"],))
                conn.execute(
                    "INSERT INTO fts_chunks (chunk_id, tenant_id, source_document, chunk_text) VALUES (?, ?, ?, ?)",
                    (chunk["chunk_id"], chunk.get("tenant_id", ""), chunk.get("source_document", ""), chunk.get("chunk_text", ""))
                )
            conn.commit()

    def search_fts5(self, query: str, tenant_id: str, limit: int = 20) -> tuple[List[Dict[str, Any]], bool]:
        """Searches the FTS5 table and returns matching chunks."""
        cleaned = re.sub(r"[^\w\s\?\&+\-]", " ", query.lower()).strip()
        if not cleaned:
            return [], False
        
        terms = [t for t in cleaned.split() if t]
        if not terms:
            return [], False

        # Default AND query for better precision
        fts_query_and = " ".join(f'"{term}"' for term in terms)
        fts_query_or = " OR ".join(f'"{term}"' for term in terms)

        def run_query(conn, q):
            results = []
            cursor = conn.execute(
                "SELECT chunk_id, source_document, chunk_text, bm25(fts_chunks) as score FROM fts_chunks WHERE fts_chunks MATCH ? AND tenant_id = ? ORDER BY score ASC LIMIT ?",
                (q, tenant_id, limit)
            )
            for row in cursor.fetchall():
                score = -row["score"]
                # Only include if score meets minimum threshold (prevent noisy 1-term matches in OR)
                if score > 0.0:
                    results.append({
                        "chunk_id": row["chunk_id"],
                        "source_document": row["source_document"],
                        "chunk_text": row["chunk_text"],
                        "tenant_id": tenant_id,
                        "score": score
                    })
            return results

        with self._get_conn() as conn:
            results = run_query(conn, fts_query_and)
            fallback_used = False
            
            if len(results) == 0:
                results = run_query(conn, fts_query_or)
                fallback_used = True
                
        return results, fallback_used
