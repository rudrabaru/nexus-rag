"""
Request body limits, applied before any parsing or authentication.

Without them the server buffers whatever a client sends: multipart bodies are parsed before the
route's auth dependency runs, so an anonymous caller could make the process hold gigabytes. A
declared Content-Length over the limit is refused at once; a body that streams past the limit
(chunked transfer, or a lying Content-Length) is cut off as it arrives.
"""
import json
from typing import Mapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyTooLarge(Exception):
    pass


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
            await self._refuse(send, limit)
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
                    await self._refuse(send, limit)
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
    async def _refuse(send: Send, limit: int) -> None:
        body = json.dumps({"detail": f"Request body exceeds the {limit // 1024} KB limit for this endpoint."}).encode()
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
