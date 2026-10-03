from typing import Any, Dict, Optional

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from src.db.rows import utcnow
from src.db.schema import workspace_settings


class WorkspaceSettingsStore:
    """What a workspace has chosen: the retrieval configuration its chat runs. No row means the environment's defaults."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def get_retrieval(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(workspace_settings.c.retrieval_config).where(workspace_settings.c.tenant_id == tenant_id)
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one_or_none()

    def put_retrieval(self, tenant_id: str, retrieval_config: Dict[str, Any]) -> None:
        stmt = insert(workspace_settings).values(
            tenant_id=tenant_id, retrieval_config=retrieval_config, updated_at=utcnow()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[workspace_settings.c.tenant_id],
            set_={"retrieval_config": stmt.excluded.retrieval_config, "updated_at": stmt.excluded.updated_at},
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def clear_retrieval(self, tenant_id: str) -> bool:
        with self._engine.begin() as conn:
            return conn.execute(delete(workspace_settings).where(workspace_settings.c.tenant_id == tenant_id)).rowcount > 0
