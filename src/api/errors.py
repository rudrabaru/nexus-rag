"""
How the API reports unexpected failures: a generic message and a reference id, never the exception.

Exception text can carry SQL, bound parameters, hostnames and provider responses. Clients get
only an id they can quote; the details stay in the server log under the same id.
"""
import logging
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from src.services.errors import InvalidRequest, PayloadTooLarge, QuotaExceeded, ServiceError, Unavailable

logger = logging.getLogger(__name__)


def new_reference() -> str:
    return uuid.uuid4().hex[:12]


def internal_error(context: str, error: BaseException) -> HTTPException:
    """Logs the failure with a reference id and returns the 500 to raise."""
    reference = new_reference()
    logger.error(f"{context} | reference={reference}", exc_info=error)
    return HTTPException(status_code=500, detail=f"Internal server error (reference {reference}).")


async def unhandled_exception_handler(request: Request, error: Exception) -> JSONResponse:
    reference = new_reference()
    logger.error(f"Unhandled error on {request.method} {request.url.path} | reference={reference}", exc_info=error)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error (reference {reference})."})


# What each kind of refusal means over HTTP. Services raise these without knowing about status codes.
SERVICE_ERROR_STATUS = {InvalidRequest: 400, PayloadTooLarge: 413, QuotaExceeded: 429, Unavailable: 503}


async def service_error_handler(request: Request, error: ServiceError) -> JSONResponse:
    status = next((code for kind, code in SERVICE_ERROR_STATUS.items() if isinstance(error, kind)), 400)
    return JSONResponse(status_code=status, content={"detail": error.message})
