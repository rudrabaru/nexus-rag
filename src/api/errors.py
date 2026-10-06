"""
The API's error contract: every non-2xx response has the same body,

    {"code": "quota_exceeded", "message": "...", "request_id": "..."}

`code` is stable and machine-readable (a client switches on it), `message` is for a person, and
`request_id` is the same id the response carries in X-Request-ID and the server log carries, so a
reported failure can be found. Unexpected errors say only that something went wrong: exception
text can carry SQL, bound parameters, hostnames and provider responses, and stays in the log.
"""
import logging
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.llm.errors import GenerationError
from src.services.errors import InvalidRequest, PayloadTooLarge, QuotaExceeded, ServiceError, Unavailable

logger = logging.getLogger(__name__)

GENERATION_FAILED_MESSAGE = "The language model could not produce an answer right now. Please try again shortly."


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None
    details: Optional[list] = None  # field-level problems of a request that failed validation


COMMON_ERRORS: Dict[int | str, Dict[str, Any]] = {
    401: {"model": ErrorBody, "description": "A valid credential is required."},
    429: {"model": ErrorBody, "description": "A rate limit or quota was reached."},
    500: {"model": ErrorBody, "description": "Unexpected failure; quote request_id."},
}

STATUS_CODES = {
    400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    413: "payload_too_large", 422: "validation_failed", 429: "rate_limited", 500: "internal_error",
    502: "generation_failed", 503: "unavailable",
}
SERVICE_ERROR_STATUS = {InvalidRequest: 400, PayloadTooLarge: 413, QuotaExceeded: 429, Unavailable: 503}


def request_id_of(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)


def error_response(
    request: Request, status: int, code: str, message: str, headers: Optional[dict] = None, details: Optional[list] = None
) -> JSONResponse:
    request_id = request_id_of(request)
    body = ErrorBody(code=code, message=message, request_id=request_id, details=details)
    # An unexpected error is answered by the outermost layer, outside the request-id middleware, so the header is set here too.
    headers = {**(headers or {}), **({"X-Request-ID": request_id} if request_id else {})}
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True), headers=headers)


async def http_exception_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
    return error_response(
        request, error.status_code, STATUS_CODES.get(error.status_code, "error"), str(error.detail), dict(error.headers or {})
    )


async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    details = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in error.errors()]
    return error_response(request, 422, "validation_failed", "The request is not valid.", details=details)


async def rate_limit_handler(request: Request, error: RateLimitExceeded) -> JSONResponse:
    return error_response(request, 429, "rate_limited", "Too many requests. Slow down and retry.", {"Retry-After": "60"})


async def service_error_handler(request: Request, error: ServiceError) -> JSONResponse:
    status = next((code for kind, code in SERVICE_ERROR_STATUS.items() if isinstance(error, kind)), 400)
    return error_response(request, status, error.code, error.message)


async def generation_error_handler(request: Request, error: GenerationError) -> JSONResponse:
    logger.error(f"Generation failed | request_id={request_id_of(request)}: {error}")
    return error_response(request, 502, "generation_failed", GENERATION_FAILED_MESSAGE)


async def unhandled_exception_handler(request: Request, error: Exception) -> JSONResponse:
    logger.error(
        f"Unhandled error on {request.method} {request.url.path} | request_id={request_id_of(request)}", exc_info=error
    )
    return error_response(request, 500, "internal_error", "Internal server error.")


def register(app) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(GenerationError, generation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
