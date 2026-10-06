"""
Optional error reporting to Sentry: unhandled exceptions only, with no personal or request data.

Off unless SENTRY_DSN is set. Nothing that identifies a caller or carries a question or an answer
is sent: no request headers (they hold the API key), body, query string, cookies, user or local
variables. What remains is the exception, its stack and the request id tag, which is enough to
find the matching log lines.
"""
import logging
from typing import Any, Dict, Optional

import sentry_sdk

logger = logging.getLogger(__name__)

_REQUEST_FIELDS_DROPPED = ("headers", "cookies", "data", "query_string", "env")


def scrub_event(event: Dict[str, Any], hint: Optional[dict] = None) -> Dict[str, Any]:
    request = event.get("request")
    if isinstance(request, dict):
        for field in _REQUEST_FIELDS_DROPPED:
            request.pop(field, None)
    event.pop("user", None)
    return event


def init_error_reporting(dsn: str, environment: str = "local") -> bool:
    """Starts reporting when a DSN is configured. Returns whether it is on."""
    if not dsn:
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        send_default_pii=False,
        traces_sample_rate=0.0,
        max_request_body_size="never",
        include_local_variables=False,
        before_send=scrub_event,
    )
    logger.info(f"Error reporting is on (environment {environment})")
    return True
