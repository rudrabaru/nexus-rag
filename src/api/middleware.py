"""
Request-level middleware, applied before routing: a request id on every request and response, and
body limits that apply before any parsing or authentication.

Without body limits the server buffers whatever a client sends: multipart bodies are parsed before
the route's auth dependency runs, so an anonymous caller could make the process hold gigabytes. A
declared Content-Length over the limit is refused at once; a body that streams past the limit
(chunked transfer, or a lying Content-Length) is cut off as it arrives.
"""
import json
import re
import uuid
from typing import Mapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-request-id"
_ACCEPTABLE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class BodyTooLarge(Exception):
    pass


class RequestIdMiddleware:
    """Gives each request an id: the caller's X-Request-ID when it is well-formed, else a new one. It is echoed in the response."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = dict(scope["headers"]).get(REQUEST_ID_HEADER, b"").decode("latin-1")
        request_id = supplied if _ACCEPTABLE_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                if not any(name.lower() == REQUEST_ID_HEADER for name, _ in headers):
                    headers.append((REQUEST_ID_HEADER, request_id.encode()))
            await send(message)

        await self.app(scope, receive, send_with_id)


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
