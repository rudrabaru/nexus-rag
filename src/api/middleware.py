"""
Request-level middleware, applied before routing: a request id and an access line for every request, and
body limits that apply before any parsing or authentication.

Without body limits the server buffers whatever a client sends: multipart bodies are parsed before
the route's auth dependency runs, so an anonymous caller could make the process hold gigabytes. A
declared Content-Length over the limit is refused at once; a body that streams past the limit
(chunked transfer, or a lying Content-Length) is cut off as it arrives.
"""
import json
import logging
import re
import time
import uuid
from typing import Mapping

import sentry_sdk
import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-request-id"
PROBE_PATHS = ("/health", "/ready")
access_log = structlog.get_logger("access")
_ACCEPTABLE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class BodyTooLarge(Exception):
    pass


class RequestContextMiddleware:
    """
    Gives each request an id (the caller's X-Request-ID when it is well-formed, else a new one),
    echoes it in the response, binds it to the logging context so every log line written while
    serving the request carries it, tags error reports with it, and writes one access line.

    The access line has the method, the path (never the query string, which can carry tokens), the
    status and the duration. Probes are logged at debug level: a platform polls them every few seconds.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = dict(scope["headers"]).get(REQUEST_ID_HEADER, b"").decode("latin-1")
        request_id = supplied if _ACCEPTABLE_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        sentry_sdk.set_tag("request_id", request_id)

        status = 500  # what the caller sees if the app dies before it starts a response
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = message.setdefault("headers", [])
                if not any(name.lower() == REQUEST_ID_HEADER for name, _ in headers):
                    headers.append((REQUEST_ID_HEADER, request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            level = logging.DEBUG if scope["path"] in PROBE_PATHS else logging.INFO
            access_log.log(
                level, "request", method=scope["method"], path=scope["path"], status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            structlog.contextvars.clear_contextvars()


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, default_limit: int, path_limits: Mapping[str, int] = None):
        self.app = app
        self.default_limit = default_limit
        self.path_limits = dict(path_limits or {})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.path_limits.get(scope["path"], self.default_limit)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._refuse(scope, send, limit)
            return

        received = 0
        refused = False

        async def limited_receive() -> Message:
            nonlocal received, refused
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit and not refused:
                    # Answered here, not by unwinding: frameworks that read the body treat a
                    # failure while reading it as a malformed request and would reply 400.
                    refused = True
                    await self._refuse(scope, send, limit)
                if refused:
                    raise BodyTooLarge()
            return message

        async def guarded_send(message: Message) -> None:
            if not refused:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except BodyTooLarge:
            pass

    @staticmethod
    async def _refuse(scope: Scope, send: Send, limit: int) -> None:
        body = json.dumps({
            "code": "payload_too_large",
            "message": f"The request body exceeds the {limit // 1024} KB limit for this endpoint.",
            "request_id": scope.get("state", {}).get("request_id"),
        }).encode()
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
