"""
The FastAPI application, assembled. create_app() has no side effects beyond building the app: the
environment is loaded by the entry point (src/api/main.py) before it is called, and the database is
not touched until the lifespan runs.

Every resource is under /v1. Breaking changes to a response shape get /v2; the generated OpenAPI
document (openapi.json, checked by a test) is the contract a client is generated from.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import errors
from src.api.lifespan import lifespan
from src.api.middleware import BodyLimitMiddleware, RequestContextMiddleware
from src.api.rate_limit import limiter
from src.api.routes import admin, chat, documents, experiments, health, workspace
from src.config import get_settings
from src.services.uploads import MAX_UPLOAD_BYTES

# 256 KB covers the largest valid JSON request (a 2,000-character query plus 20 chat turns); an
# upload may carry MAX_UPLOAD_BYTES plus a little multipart framing.
MAX_JSON_BODY_BYTES = 256 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

DESCRIPTION = (
    "Nexus: retrieval-augmented question answering over your own documents, with the tools to measure how well it "
    "retrieves. Authenticate with `X-API-Key` (a workspace key); `/v1/admin` takes `X-Admin-Key`. Every error has "
    "the body `{code, message, request_id}`."
)


def create_app() -> FastAPI:
    app = FastAPI(title="Nexus RAG API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan, responses=errors.COMMON_ERRORS)

    # add_middleware wraps: the last one added is outermost. Request ids come first so every
    # response, including a refusal by the layers below, carries one.
    app.add_middleware(
        BodyLimitMiddleware,
        default_limit=MAX_JSON_BODY_BYTES,
        path_limits={"/v1/documents": MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES},
    )
    # CORS: no origins are allowed unless ALLOWED_ORIGINS lists them. Credentials are never
    # allowed cross-origin because authentication uses headers, not cookies.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RequestContextMiddleware)

    app.state.limiter = limiter
    errors.register(app)

    for router in (health.router, chat.router, documents.router, workspace.router, experiments.router, admin.router):
        app.include_router(router)
    return app
