"""Read-only access to the legacy stores: the Qdrant collection and the SQLite registry."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient

SCROLL_PAGE_SIZE = 256


def read_qdrant_points(url: str, api_key: str, collection: str) -> List[Dict[str, Any]]:
    """Every point with its payload and vector. The collection uses one unnamed vector per point."""
    client = QdrantClient(url=url, api_key=api_key, timeout=120.0)
    points, offset = [], None
    while True:
        page, offset = client.scroll(
            collection_name=collection, limit=SCROLL_PAGE_SIZE, with_payload=True, with_vectors=True, offset=offset
        )
        for point in page:
            if not isinstance(point.vector, list):
                raise ValueError(f"Point {point.id} has named vectors; this migration expects one unnamed vector.")
            points.append({"payload": point.payload or {}, "vector": point.vector})
        if offset is None:
            return points


def parse_json(value: Optional[str], default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def read_registry(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Rows of the SQLite registry, opened read-only. Missing tables read as empty."""
    if not path.exists():
        return {"documents": [], "jobs": [], "api_keys": [], "query_logs": []}

    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def rows(table: str) -> List[Dict[str, Any]]:
        try:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        except sqlite3.OperationalError:
            return []

    try:
        documents = rows("documents")
        for d in documents:
            d["chunk_ids"] = parse_json(d.get("chunk_ids"), [])
            d["stats"] = parse_json(d.get("stats"), {})
        jobs = rows("jobs")
        for j in jobs:
            j["metadata"] = parse_json(j.get("metadata"), None)
        query_logs = rows("observability_logs")
        for q in query_logs:
            q["details"] = parse_json(q.get("details"), {})
        return {"documents": documents, "jobs": jobs, "api_keys": rows("api_keys"), "query_logs": query_logs}
    finally:
        conn.close()
