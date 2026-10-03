from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

# A worker beats every 10 seconds; three missed beats is the same threshold the recovery sweep uses.
WORKER_ALIVE_SECONDS = 30


class SystemStore:
    """What the service can tell about its own surroundings: is the database reachable, is a worker running."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def database_ok(self) -> bool:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def workers_online(self) -> int:
        """Workers whose heartbeat is recent. Workers run on demand, so zero is normal: jobs wait in the queue."""
        since = datetime.now(timezone.utc) - timedelta(seconds=WORKER_ALIVE_SECONDS)
        stmt = text("SELECT count(*) FROM procrastinate_workers WHERE last_heartbeat > :since")
        with self._engine.connect() as conn:
            return int(conn.execute(stmt, {"since": since}).scalar_one())
