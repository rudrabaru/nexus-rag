import sys
import asyncio
import logging
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException

# override=False: variables already set in the real environment win over the file.
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=False)

from src.api.startup import lifespan
from src.api.routes import query, ingest, documents, admin
from src.config import get_settings
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.api.routes.ingest import limiter as ingest_limiter

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Nexus RAG API", lifespan=lifespan)

# CORS: no origins are allowed unless ALLOWED_ORIGINS lists them. Credentials are
# never allowed cross-origin because authentication uses headers, not cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
