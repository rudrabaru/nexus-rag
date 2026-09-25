"""
The worker process: runs ingestion jobs from the Postgres-backed queue.

Run it with Procrastinate's own CLI, which handles signal-driven graceful shutdown so an
in-flight ingestion is not cut off mid-write:

    procrastinate --app=src.jobs.worker.app worker --queues=ingest

`python -m src.jobs.worker` is a thin convenience wrapper around the same call, for local
development without the CLI.

This module is the one place worker-only dependencies (the dispatcher, MarkItDown, the
embedding client) are allowed to be imported at process start, through src.jobs.tasks. The
API process never imports this module.
"""
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from procrastinate import App, PsycopgConnector

# The worker is its own process (never uvicorn), and PsycopgConnector's async pool refuses to
# run under Windows' default ProactorEventLoop ("Psycopg cannot use the 'ProactorEventLoop' to
# run in async mode"). This only affects local Windows development; the worker runs on Linux
# in every deployed environment. src/api/main.py sets the opposite policy for its own process.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# override=False: variables already set in the real environment win over the file, matching
# src/api/main.py's rule.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=False)

from src.config import get_settings  # noqa: E402
from src.jobs.contract import INGEST_QUEUE, TASK_NAMESPACE  # noqa: E402
from src.jobs.tasks import blueprint  # noqa: E402
from src.registry.engine import get_sync_engine, libpq_url  # noqa: E402
from src.registry.schema_version import assert_schema_current  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_settings = get_settings()
_missing = _settings.missing_required()
if _missing:
    raise RuntimeError(f"Missing or invalid configuration: {', '.join(_missing)}. Worker cannot start.")
assert_schema_current(get_sync_engine())

app = App(connector=PsycopgConnector(conninfo=libpq_url(_settings.database_url)))
app.add_tasks_from(blueprint, namespace=TASK_NAMESPACE)


async def _run() -> None:
    async with app.open_async():
        logger.info(f"Worker ready. queue={INGEST_QUEUE} concurrency={_settings.worker_concurrency}")
        await app.run_worker_async(queues=[INGEST_QUEUE], concurrency=_settings.worker_concurrency)


if __name__ == "__main__":
    asyncio.run(_run())
