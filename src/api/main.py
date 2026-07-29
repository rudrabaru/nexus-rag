import sys
import asyncio
import logging
import os
import uuid
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, Query, Form, HTTPException

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

from src.api.startup import lifespan
from src.api.routes import query, ingest, documents, admin
from src.api.dependencies import get_auth_store
from src.registry.auth_store import AuthStore
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.api.routes.ingest import limiter as ingest_limiter

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Nexus RAG API", lifespan=lifespan)

# CORS
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting — each router owns its Limiter; app state needs one for the handler
app.state.limiter = ingest_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(admin.router)
app.include_router(
    documents.router,
    prefix="/documents",
    tags=["documents"],
)
@app.get("/health")
def health_check(request: Request):
    # Liveness probe: returns immediately if server is up
    if hasattr(request.app.state, "init_error"):
        return {"status": "error", "message": request.app.state.init_error}
    return {"status": "ok"}


@app.get("/ready")
def ready_check(request: Request):
    # Readiness probe: returns OK only when models are loaded
    if hasattr(request.app.state, "init_error"):
        return {"status": "error", "message": request.app.state.init_error}

    if getattr(request.app.state, "ready", False):
        generator = request.app.state.generator
        return {"status": "ready", "provider": generator.config.provider}
    raise HTTPException(status_code=503, detail="Models still loading")


@app.post("/register")
def register_tenant(
    secret: str = Form(None),
    auth_store: AuthStore = Depends(get_auth_store)
):
    import secrets
    expected = os.environ.get("REGISTRATION_SECRET")
    if expected:
        if not secret or not secrets.compare_digest(secret, expected):
            raise HTTPException(status_code=403, detail="Invalid registration secret")
    tenant_id = str(uuid.uuid4())
    api_key = auth_store.create_api_key(tenant_id)
    return {"api_key": api_key, "tenant_id": tenant_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
