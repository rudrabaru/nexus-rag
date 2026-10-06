from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def row_to_dict(row) -> Optional[Dict[str, Any]]:
    """
    Converts a result row to a plain dict. Timestamps become ISO-8601 strings so callers and
    API responses keep the shape they had with the SQLite registry.
    """
    if row is None:
        return None
    return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in row._mapping.items()}
