"""
The FastAPI application, assembled. create_app() has no side effects beyond building the app: the
environment is loaded by the entry point (src/api/main.py) before it is called, and the
database is not touched until the lifespan runs.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.errors import service_error_handler, unhandled_exception_handler
from src.api.middleware import BodyLimitMiddleware
from src.api.rate_limit import limiter
from src.api.routes import admin, documents, health, ingest, query
from src.api.startup import lifespan
from src.config import get_settings
from src.services.errors import ServiceError
from src.services.ingestion_service import MAX_UPLOAD_BYTES

# 256 KB covers the largest valid JSON request (a 2,000-character query plus 20 chat turns); an
# upload may carry MAX_UPLOAD_BYTES plus a little multipart framing.
MAX_JSON_BODY_BYTES = 256 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


def create_app() -> FastAPI:
    app = FastAPI(title="Nexus RAG API", lifespan=lifespan)

    app.add_middleware(
        BodyLimitMiddleware,
        default_limit=MAX_JSON_BODY_BYTES,
        path_limits={"/ingest": MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES},
    )
    # CORS: no origins are allowed unless ALLOWED_ORIGINS lists them. Credentials are never
    # allowed cross-origin because authentication uses headers, not cookies.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(admin.router)
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    return app
