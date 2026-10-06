"""
Retention: the tables that grow with traffic are pruned by age, so Neon's 1 GB free plan is not
slowly filled by logs.

Nothing here runs on a timer. The free API host sleeps and the workers run on demand, so a
scheduler would rarely be awake. Instead the pruning is idempotent, cheap (the columns are
indexed) and runs whenever a process starts: the API at boot, a worker at launch.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import delete
from sqlalchemy.engine import Engine

from src.db.schema import fetch_log, pipeline_events, query_logs

logger = logging.getLogger(__name__)

# Table -> (age column, days kept). Operational choices, not corpus-tuned:
# - pipeline_events: several rows per query and per job; only useful while debugging recent behaviour.
# - query_logs: the per-query cost and latency history a dashboard shows; small rows.
# - fetch_log: the quota needs 24 hours, but the audit trail is what makes abuse attributable.
RETENTION_DAYS = {
    "pipeline_events": (pipeline_events, pipeline_events.c.timestamp, 14),
    "query_logs": (query_logs, query_logs.c.timestamp, 90),
    "fetch_log": (fetch_log, fetch_log.c.created_at, 90),
}
FINISHED_QUEUE_JOB_HOURS = 24 * 30  # failed jobs stay a month for inspection; successful ones are deleted on completion


def prune(engine: Engine, now: Optional[datetime] = None) -> Dict[str, int]:
    """Deletes rows older than each table's retention. Returns how many were deleted per table."""
    now = now or datetime.now(timezone.utc)
    deleted = {}
    with engine.begin() as conn:
        for name, (table, column, days) in RETENTION_DAYS.items():
            deleted[name] = conn.execute(delete(table).where(column < now - timedelta(days=days))).rowcount
    if any(deleted.values()):
        logger.info(f"Retention pruned {deleted}")
    return deleted
