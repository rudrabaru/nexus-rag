import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.container import build_components
from src.config import get_settings
from src.config_checks import config_problems
from src.jobs.queue import api_queue
from src.observability.logger import PipelineLogger
from src.services.ingestion_service import IngestionService
from src.stores.api_keys import AuthStore
from src.stores.documents import DocumentStore
from src.stores.fetches import FetchStore
from src.stores.jobs import JobStore
from src.stores.system import SystemStore
from src.stores.workspace import WorkspaceSettingsStore
from src.db.engine import dispose_engines, get_sync_engine
from src.stores.query_log import QueryLogStore
from src.db.schema_version import assert_schema_current
from src.maintenance import prune

logger = logging.getLogger(__name__)


def _initialize(app: FastAPI) -> None:
    """Blocking initialisation, run in a worker thread so the server binds its port immediately."""
    settings = get_settings()
    sync_engine = get_sync_engine()
    assert_schema_current(sync_engine)
    try:
        prune(sync_engine)
    except Exception:  # housekeeping must never stop the API from starting
        logger.exception("Retention pruning failed")

    components = build_components()
    app.state.retrieval = components.retrieval
    app.state.generator = components.generator
    app.state.evaluator = components.evaluator
    app.state.rewriter = components.rewriter
    app.state.auth_store = AuthStore(sync_engine)
    app.state.query_log = QueryLogStore(sync_engine)
    app.state.pipeline_logger = PipelineLogger("nexus_rag", engine=sync_engine)

    # The API only defers ingestion jobs; it never runs them (src/jobs/workers.py does), so a
    # crashed or restarted API process cannot leave a job stuck "processing" — the worker's
    # own stalled-job detection (heartbeats) is what recovers those.
    app.state.job_queue = api_queue(settings.database_url.get_secret_value()).open()
    app.state.documents = DocumentStore(sync_engine)
    app.state.workspace = WorkspaceSettingsStore(sync_engine)
    app.state.system = SystemStore(sync_engine)
    app.state.jobs = JobStore(sync_engine)
    app.state.ingestion = IngestionService(
        app.state.job_queue, app.state.documents, app.state.jobs, FetchStore(sync_engine), settings
    )

    index_id = components.retrieval.default_index_id
    index_size = components.retrieval.chunk_store().get_collection_size()
    logger.info(
        f"RAG Pipeline API ready. Chat model: {components.model}, "
        f"index: {index_id} ({index_size} chunks)"
    )
    if index_size == 0:
        logger.warning(
            f"Index {index_id} holds no chunks, so every search returns nothing. "
            "Ingest documents with EMBEDDING_PROVIDER set to the model this index should use."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    problems = config_problems(settings, "api")
    if problems:
        raise RuntimeError(f"Missing or invalid configuration: {', '.join(problems)}. Startup aborted.")

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
