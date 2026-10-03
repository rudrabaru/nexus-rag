"""
Worker launcher: one command for both workers.

    python -m src.jobs.workers ingest            # parse worker: uploads and fetched pages -> chunks and vectors
    python -m src.jobs.workers fetch             # fetch worker: web pages and sitemaps, through reader APIs
    python -m src.jobs.workers ingest --drain    # run what is queued, then exit

--drain is how workers run on a laptop: nothing polls the database while nobody is working, which
keeps Neon's compute asleep (its free plan has a monthly compute-hour budget). Jobs wait in the
queue while no worker runs; queries and stored results never depend on one.

Each worker holds exactly the tasks of its own queue (asserted in build_app): the parse worker
runs on a machine that must never contact a website, so it must not be able to run a fetch task
even by misconfiguration, and the slim fetch worker must not carry parsing code, so a queue's
tasks are imported only when that queue is started.
"""
import argparse
import asyncio
import importlib
import logging
from typing import List, Optional

from procrastinate import App, Blueprint, PsycopgConnector

from src.config import get_settings
from src.jobs.contract import FETCH_QUEUE, INGEST_QUEUE, TASK_NAMESPACE
from src.registry.engine import libpq_url
from src.runtime import bootstrap

logger = logging.getLogger(__name__)

# queue -> (module holding its tasks, the Blueprint in it)
TASK_MODULES = {
    INGEST_QUEUE: ("src.jobs.ingest_tasks", "blueprint"),
    FETCH_QUEUE: ("src.jobs.fetch_tasks", "fetch_blueprint"),
}
FETCH_CONCURRENCY = 1  # per-domain pacing is kept per process, and keyless Jina Reader allows ~20 requests a minute per IP


def build_app(blueprint: Blueprint, queue: str) -> App:
    """A Procrastinate App holding exactly the tasks of one queue. The connection opens later, not here."""
    app = App(connector=PsycopgConnector(conninfo=libpq_url(get_settings().database_url.get_secret_value())))
    app.add_tasks_from(blueprint, namespace=TASK_NAMESPACE)
    # Only our own tasks count: every App also registers Procrastinate's builtin cleanup task
    # (remove_old_jobs) on its own queue, which contacts nothing but the database.
    foreign = sorted(
        name for name, task in app.tasks.items() if name.startswith(f"{TASK_NAMESPACE}:") and task.queue != queue
    )
    if foreign:
        raise RuntimeError(f"The {queue!r} worker must only run {queue!r} tasks; found {foreign}.")
    return app


async def run(app: App, queue: str, concurrency: int, drain: bool = False) -> None:
    async with app.open_async():
        logger.info(f"Worker ready. queue={queue} concurrency={concurrency} drain={drain}")
        # delete_jobs: a successful job's queue row is removed; failures stay for inspection.
        await app.run_worker_async(queues=[queue], concurrency=concurrency, wait=not drain, delete_jobs="successful")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue", choices=sorted(TASK_MODULES))
    parser.add_argument("--drain", action="store_true", help="exit when the queue is empty instead of waiting for jobs")
    args = parser.parse_args(argv)

    settings = bootstrap("worker", async_postgres_driver=True)
    module_name, blueprint_name = TASK_MODULES[args.queue]
    blueprint = getattr(importlib.import_module(module_name), blueprint_name)
    concurrency = settings.worker_concurrency if args.queue == INGEST_QUEUE else FETCH_CONCURRENCY
    asyncio.run(run(build_app(blueprint, args.queue), args.queue, concurrency, args.drain))


if __name__ == "__main__":
    main()
