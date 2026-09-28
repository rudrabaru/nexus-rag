"""
Shared start-up for the two worker processes (src/jobs/worker.py, src/jobs/fetch_worker.py):
environment, configuration and schema checks, and a Procrastinate App holding exactly the
tasks of one queue.

Queue isolation is asserted, not assumed: the parse worker runs on hosting that must never
send requests to third-party sites (the Hugging Face Space), so it must not be able to run a
fetch task even by misconfiguration, and the slim fetch worker must not carry parsing code.
"""
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from procrastinate import App, Blueprint, PsycopgConnector

from src.config import get_settings
from src.jobs.contract import TASK_NAMESPACE
from src.registry.engine import get_sync_engine, libpq_url
from src.registry.schema_version import assert_schema_current

logger = logging.getLogger(__name__)


def prepare_process() -> None:
    # A worker is its own process (never uvicorn), and PsycopgConnector's async pool refuses
    # Windows' default ProactorEventLoop. Local Windows development only; deployments are Linux.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # override=False: variables already set in the real environment win over the file.
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    missing = get_settings().missing_required()
    if missing:
        raise RuntimeError(f"Missing or invalid configuration: {', '.join(missing)}. Worker cannot start.")
    assert_schema_current(get_sync_engine())


def build_app(blueprint: Blueprint, queue: str) -> App:
    app = App(connector=PsycopgConnector(conninfo=libpq_url(get_settings().database_url)))
    app.add_tasks_from(blueprint, namespace=TASK_NAMESPACE)
    # Only our own tasks count: every App also registers Procrastinate's builtin cleanup task
    # (remove_old_jobs) on its own queue, which contacts nothing but the database.
    foreign = sorted(
        name for name, task in app.tasks.items() if name.startswith(f"{TASK_NAMESPACE}:") and task.queue != queue
    )
    if foreign:
        raise RuntimeError(f"The {queue!r} worker must only run {queue!r} tasks; found {foreign}.")
    return app


async def run(app: App, queue: str, concurrency: int) -> None:
    async with app.open_async():
        logger.info(f"Worker ready. queue={queue} concurrency={concurrency}")
        await app.run_worker_async(queues=[queue], concurrency=concurrency)
