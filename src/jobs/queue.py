"""
The durable job queue: Procrastinate, on the same Postgres as everything else.

Why Procrastinate: jobs survive restarts and crashes, retries and per-document locks are
built in, and it needs no broker beyond the database the platform already runs (Celery or
RQ would add Redis).

The API only defers jobs. It uses Procrastinate's sync connector from a worker thread
(asyncio.to_thread), the same pattern as the registry, and never imports task code; tasks
are referenced by name (src/jobs/contract.py). The worker builds its own App with the async
connector and the task implementations (src/jobs/worker.py).
"""
import procrastinate

from src.registry.engine import libpq_url

API_POOL_SIZE = 2  # the API only inserts jobs; each defer holds a connection for one statement


def api_queue(database_url: str) -> procrastinate.App:
    connector = procrastinate.SyncPsycopgConnector(
        conninfo=libpq_url(database_url), min_size=1, max_size=API_POOL_SIZE
    )
    return procrastinate.App(connector=connector)
