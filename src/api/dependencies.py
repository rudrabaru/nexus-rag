import asyncio
from fastapi import HTTPException, Request
from typing import Any, Optional

from src.generating.generator import RAGGenerator
from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.query_rewriter import QueryRewriter
from src.retrieving.retriever import OptionalReranker
from src.registry.database import DocumentRegistry
from src.registry.auth_store import AuthStore
from src.registry.metrics_store import MetricsStore
def _check_ready(request: Request):
    """Helper to check if the background task crashed or is still initializing."""
    # If ready=True, always serve — a prior partial error is irrelevant.
    if getattr(request.app.state, "ready", False):
        return
    # Not ready yet: distinguish between a fatal crash and still-initializing.
    if hasattr(request.app.state, "init_error"):
        raise HTTPException(status_code=500, detail=request.app.state.init_error)
    raise HTTPException(status_code=503, detail="Service is initializing, please wait...")

def get_generator(request: Request) -> RAGGenerator:
    _check_ready(request)
    return request.app.state.generator

def get_retriever(request: Request) -> Any:
    _check_ready(request)
    return request.app.state.retriever

def get_reranker(request: Request) -> OptionalReranker:
    _check_ready(request)
    return request.app.state.reranker

def get_evaluator(request: Request) -> FaithfulnessEvaluator:
    _check_ready(request)
    return request.app.state.evaluator

def get_rewriter(request: Request) -> Optional[QueryRewriter]:
    _check_ready(request)
    return getattr(request.app.state, "rewriter", None)

def get_registry(request: Request) -> DocumentRegistry:
    _check_ready(request)
    return request.app.state.registry

def get_auth_store(request: Request) -> AuthStore:
    _check_ready(request)
    return request.app.state.auth_store

def get_metrics_store(request: Request) -> MetricsStore:
    _check_ready(request)
    return request.app.state.metrics_store

def get_pipeline_logger(request: Request):
    return getattr(request.app.state, "pipeline_logger", None)

def get_ingestion_semaphore(request: Request) -> asyncio.Semaphore:
    _check_ready(request)
    return request.app.state.ingestion_semaphore
