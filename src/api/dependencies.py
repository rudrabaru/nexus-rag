from fastapi import HTTPException, Request
from typing import Optional

import procrastinate

from src.generating.generator import RAGGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter
from src.retrieving.pipeline import RetrievalResources
from src.services.ingestion_service import IngestionService
from src.stores.api_keys import AuthStore
from src.stores.documents import DocumentStore
from src.stores.jobs import JobStore


def _check_ready(request: Request):
    """Helper to check if the background task crashed or is still initializing."""
    # If ready=True, always serve — a prior partial error is irrelevant.
    if getattr(request.app.state, "ready", False):
        return
    # Not ready yet: distinguish between a fatal crash and still-initializing.
    if hasattr(request.app.state, "init_error"):  # the reason is in the server log, not in the response
        raise HTTPException(status_code=503, detail="Service unavailable.")
    raise HTTPException(status_code=503, detail="Service is starting up, please retry shortly.")

def get_generator(request: Request) -> RAGGenerator:
    _check_ready(request)
    return request.app.state.generator

def get_retrieval(request: Request) -> RetrievalResources:
    _check_ready(request)
    return request.app.state.retrieval

def get_evaluator(request: Request) -> FaithfulnessEvaluator:
    _check_ready(request)
    return request.app.state.evaluator

def get_rewriter(request: Request) -> Optional[QueryRewriter]:
    _check_ready(request)
    return getattr(request.app.state, "rewriter", None)

def get_documents(request: Request) -> DocumentStore:
    _check_ready(request)
    return request.app.state.documents


def get_jobs(request: Request) -> JobStore:
    _check_ready(request)
    return request.app.state.jobs


def get_ingestion_service(request: Request) -> IngestionService:
    _check_ready(request)
    return request.app.state.ingestion

def get_auth_store(request: Request) -> AuthStore:
    _check_ready(request)
    return request.app.state.auth_store

def get_pipeline_logger(request: Request):
    return getattr(request.app.state, "pipeline_logger", None)

def get_job_queue(request: Request) -> procrastinate.App:
    _check_ready(request)
    return request.app.state.job_queue
