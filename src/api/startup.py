import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.factory import _init_components
from src.config import get_settings
from src.jobs.queue import api_queue
from src.observability.logger import PipelineLogger
from src.registry.auth_store import AuthStore
from src.registry.engine import dispose_engines, get_sync_engine
from src.registry.metrics_store import MetricsStore
from src.registry.schema_version import assert_schema_current

logger = logging.getLogger(__name__)


def _initialize(app: FastAPI) -> None:
    """Blocking initialisation, run in a worker thread so the server binds its port immediately."""
    settings = get_settings()
    sync_engine = get_sync_engine()
    assert_schema_current(sync_engine)

    components = _init_components()
    app.state.chunk_store = components.chunk_store
    app.state.registry = components.registry
    app.state.retriever = components.retriever
    app.state.reranker = components.reranker
    app.state.generator = components.generator
    app.state.evaluator = components.evaluator
    app.state.rewriter = components.rewriter
    app.state.embedding_generator = components.embedding_generator
    app.state.auth_store = AuthStore(sync_engine)
    app.state.metrics_store = MetricsStore(sync_engine)
    app.state.pipeline_logger = PipelineLogger("nexus_rag", engine=sync_engine)

    # The API only defers ingestion jobs; it never runs them (src/jobs/worker.py does), so a
    # crashed or restarted API process cannot leave a job stuck "processing" — the worker's
    # own stalled-job detection (heartbeats) is what recovers those.
    app.state.job_queue = api_queue(settings.database_url).open()

    logger.info(
        f"RAG Pipeline API ready. Provider: {components.provider}, Model: {components.model_name}, "
        f"chunks indexed: {components.chunk_store.get_collection_size()}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    missing = settings.missing_required()
    if missing:
        raise RuntimeError(f"Missing or invalid configuration: {', '.join(missing)}. Startup aborted.")
    settings.warn_on_legacy_secrets()

    # The semaphore belongs to the serving event loop, so it is created here, not in the thread.
    app.state.query_semaphore = asyncio.Semaphore(settings.query_concurrency)

    async def _load():
        try:
            await asyncio.to_thread(_initialize, app)
            app.state.ready = True
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            app.state.init_error = str(e)

    app.state._startup_task = asyncio.create_task(_load())

    yield

    app.state._startup_task.cancel()
    try:
        await app.state._startup_task
    except asyncio.CancelledError:
        pass
    pipeline_logger = getattr(app.state, "pipeline_logger", None)
    if pipeline_logger:
        pipeline_logger.close()
    job_queue = getattr(app.state, "job_queue", None)
    if job_queue:
        job_queue.close()
    await dispose_engines()
