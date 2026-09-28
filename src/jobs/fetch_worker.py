"""
The fetch worker: runs fetch jobs (web pages and sitemaps, through reader APIs) and hands
each finished job to the parse worker. Slim: it runs on the API image, next to the API.

    procrastinate --app=src.jobs.fetch_worker.app worker --queues=fetch --concurrency=1

Concurrency 1 by design: per-domain pacing (FETCH_MIN_INTERVAL_SECONDS) is kept per process,
and keyless Jina Reader allows ~20 requests/min per IP, so parallel fetching would only queue
behind the same limits.
"""
import asyncio

from src.jobs.bootstrap import build_app, prepare_process, run

prepare_process()

from src.jobs.contract import FETCH_QUEUE  # noqa: E402
from src.jobs.fetch_tasks import fetch_blueprint  # noqa: E402

app = build_app(fetch_blueprint, FETCH_QUEUE)

if __name__ == "__main__":
    asyncio.run(run(app, FETCH_QUEUE, concurrency=1))
