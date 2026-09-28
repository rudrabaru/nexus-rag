"""
The parse worker: runs ingest jobs (parse, chunk, embed, commit) from the Postgres queue.
Heavy (Docling), and compute-only: it never contacts a website (src/jobs/fetch_worker.py does
the fetching, through reader APIs).

Run it with Procrastinate's CLI, which handles signal-driven graceful shutdown so an in-flight
ingestion is not cut off mid-write:

    procrastinate --app=src.jobs.worker.app worker --queues=ingest

`python -m src.jobs.worker` is a thin convenience wrapper for local development.
"""
import asyncio

from src.jobs.bootstrap import build_app, prepare_process, run

prepare_process()

from src.config import get_settings  # noqa: E402
from src.jobs.contract import INGEST_QUEUE  # noqa: E402
from src.jobs.tasks import blueprint  # noqa: E402

app = build_app(blueprint, INGEST_QUEUE)

if __name__ == "__main__":
    asyncio.run(run(app, INGEST_QUEUE, get_settings().worker_concurrency))
