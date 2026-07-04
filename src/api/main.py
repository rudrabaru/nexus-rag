import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import logging
from fastapi import FastAPI

# Add project root to path
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.api.startup import lifespan, app_state
from src.api.routes import query, ingest, documents
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="GCP RAG API", lifespan=lifespan)

# CORS
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication
api_key_header = APIKeyHeader(name="RAG-API-KEY", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = os.environ.get("RAG_API_KEY")
    if expected_api_key and api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Could not validate API KEY")
    return api_key


# Routes
app.include_router(query.router, dependencies=[Security(verify_api_key)])
app.include_router(ingest.router, dependencies=[Security(verify_api_key)])
app.include_router(
    documents.router,
    prefix="/documents",
    tags=["documents"],
    dependencies=[Security(verify_api_key)],
)


@app.get("/health")
def health_check():
    # Liveness probe: returns immediately if server is up
    if "init_error" in app_state:
        return {"status": "error", "message": app_state["init_error"]}
    return {"status": "ok"}


@app.get("/ready")
def ready_check():
    # Readiness probe: returns OK only when models are loaded
    if "init_error" in app_state:
        return {"status": "error", "message": app_state["init_error"]}

    generator = app_state.get("generator")
    retriever = app_state.get("retriever")
    registry = app_state.get("registry")
    if generator and retriever and registry:
        return {"status": "ready", "provider": generator.config.provider}
    raise HTTPException(status_code=503, detail="Models still loading")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
