import sys
import asyncio
import logging
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# override=False: variables already set in the real environment win over the file.
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=False)

from src.api.errors import unhandled_exception_handler
from src.api.middleware import BodyLimitMiddleware
from src.api.startup import lifespan
from src.services.ingestion_service import MAX_UPLOAD_BYTES
from src.api.routes import query, ingest, documents, admin
from src.config import get_settings
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.api.rate_limit import limiter

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_JSON_BODY_BYTES = 256 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

app = FastAPI(title="Nexus RAG API", lifespan=lifespan)

# Added first so it runs outermost-but-one: bodies are bounded before parsing or auth.
# 256 KB covers the largest valid JSON request (a 2,000-character query plus 20 history turns);
# an upload may carry MAX_UPLOAD_BYTES plus a little multipart framing.
app.add_middleware(
    BodyLimitMiddleware,
    default_limit=MAX_JSON_BODY_BYTES,
    path_limits={"/ingest": MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES},
)

# CORS: no origins are allowed unless ALLOWED_ORIGINS lists them. Credentials are
# never allowed cross-origin because authentication uses headers, not cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: one Limiter for every route (src/api/rate_limit.py)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

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
    # Liveness probe. A failed initialisation never recovers in-process, so it reports 503
    # and the platform restarts or replaces the instance instead of routing traffic to it.
    if hasattr(request.app.state, "init_error"):  # the reason is in the server log, not in the response
        return JSONResponse(status_code=503, content={"status": "error"})
    return {"status": "ok"}


@app.get("/ready")
def ready_check(request: Request):
    # Readiness probe: 200 only once initialisation has finished; a failed one is not ready either.
    if hasattr(request.app.state, "init_error"):
        return JSONResponse(status_code=503, content={"status": "error"})

    if getattr(request.app.state, "ready", False):
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Starting up.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
