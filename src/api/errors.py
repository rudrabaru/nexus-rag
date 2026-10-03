"""
How the API reports unexpected failures: a generic message and a reference id, never the exception.

Exception text can carry SQL, bound parameters, hostnames and provider responses. Clients get
only an id they can quote; the details stay in the server log under the same id.
"""
import logging
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

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
