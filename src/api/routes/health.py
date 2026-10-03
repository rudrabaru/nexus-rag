from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(request: Request):
    # Liveness probe. A failed initialisation never recovers in-process, so it reports 503
    # and the platform restarts or replaces the instance instead of routing traffic to it.
    if hasattr(request.app.state, "init_error"):  # the reason is in the server log, not in the response
        return JSONResponse(status_code=503, content={"status": "error"})
    return {"status": "ok"}


@router.get("/ready")
def ready_check(request: Request):
    # Readiness probe: 200 only once initialisation has finished; a failed one is not ready either.
    if hasattr(request.app.state, "init_error"):
        return JSONResponse(status_code=503, content={"status": "error"})
    if getattr(request.app.state, "ready", False):
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Starting up.")
