"""What routes receive from the running application. Everything is built once at start-up (src/api/startup.py) and kept on app.state."""
from fastapi import HTTPException, Request

import procrastinate

from src.services.chat_service import ChatService
from src.services.ingestion_service import IngestionService
from src.stores.api_keys import AuthStore
from src.stores.documents import DocumentStore
from src.stores.jobs import JobStore
from src.stores.query_log import QueryLogStore
from src.stores.system import SystemStore
from src.stores.workspace import WorkspaceSettingsStore


def _check_ready(request: Request) -> None:
    """Refuses while start-up is running or has failed. The reason for a failure is in the server log, not in the response."""
    if getattr(request.app.state, "ready", False):
        return
    if hasattr(request.app.state, "init_error"):
        raise HTTPException(status_code=503, detail="Service unavailable.")
    raise HTTPException(status_code=503, detail="Service is starting up, please retry shortly.")


def _state(request: Request, name: str):
    _check_ready(request)
    return getattr(request.app.state, name)


def get_auth_store(request: Request) -> AuthStore:
    return _state(request, "auth_store")


def get_documents(request: Request) -> DocumentStore:
    return _state(request, "documents")


def get_jobs(request: Request) -> JobStore:
    return _state(request, "jobs")


def get_query_log(request: Request) -> QueryLogStore:
    return _state(request, "query_log")


def get_workspace(request: Request) -> WorkspaceSettingsStore:
    return _state(request, "workspace")


def get_system(request: Request) -> SystemStore:
    return _state(request, "system")


def get_ingestion_service(request: Request) -> IngestionService:
    return _state(request, "ingestion")


def get_job_queue(request: Request) -> procrastinate.App:
    return _state(request, "job_queue")


def get_pipeline_logger(request: Request):
    return getattr(request.app.state, "pipeline_logger", None)


def get_chat_service(request: Request) -> ChatService:
    """Assembled per request from the long-lived parts: cheap, and every part stays replaceable in tests."""
    _check_ready(request)
    state = request.app.state
    return ChatService(
        retrieval=state.retrieval,
        generator=state.generator,
        evaluator=state.evaluator,
        rewriter=getattr(state, "rewriter", None),
        documents=getattr(state, "documents", None),
        query_log=getattr(state, "query_log", None),
        workspace=getattr(state, "workspace", None),
        events=getattr(state, "pipeline_logger", None),
    )
