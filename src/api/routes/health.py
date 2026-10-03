import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.dependencies import get_system
from src.api.rate_limit import READ_LIMIT, limiter
from src.api.security import require_tenant
from src.stores.system import SystemStore

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: str


class Workers(BaseModel):
    online: int
    detail: str


def _failed(request: Request) -> bool:
    return hasattr(request.app.state, "init_error")  # the reason is in the server log, not in the response


@router.get("/health", response_model=Health, operation_id="health")
def health(request: Request):
    """Liveness. A failed start never recovers in-process, so it reports 503 and the platform replaces the instance."""
    if _failed(request):
        return JSONResponse(status_code=503, content={"status": "error"})
    return Health(status="ok")


@router.get("/ready", response_model=Health, operation_id="ready")
async def ready(request: Request):
    """Readiness: started, and the database answers. Anything else is 503, so traffic is not routed here yet."""
    if _failed(request):
        return JSONResponse(status_code=503, content={"status": "error"})
    if not getattr(request.app.state, "ready", False):
        return JSONResponse(status_code=503, content={"status": "starting"})
    if not await asyncio.to_thread(request.app.state.system.database_ok):
        return JSONResponse(status_code=503, content={"status": "database_unreachable"})
    return Health(status="ready")


@router.get("/v1/system/workers", response_model=Workers, operation_id="workers", tags=["system"])
@limiter.limit(READ_LIMIT)
async def workers(request: Request, _: str = Depends(require_tenant), system: SystemStore = Depends(get_system)):
    """Whether a batch worker is running. Workers run on demand, so none online is normal: documents and
    experiments wait in the queue until one starts, while chat and stored results are unaffected."""
    online = await asyncio.to_thread(system.workers_online)
    detail = (
        f"{online} worker(s) are processing the queue." if online
        else "No worker is running: submitted documents wait in the queue until one is started. Chat is unaffected."
    )
    return Workers(online=online, detail=detail)
