"""
What use-cases can refuse to do, in the vocabulary of the domain rather than of HTTP.

A service raises one of these; the API layer decides the status code (src/api/errors.py). That
keeps services usable from a worker or a command-line tool, which have no HTTP.
"""


class ServiceError(Exception):
    """A request the service will not carry out. The message is safe to show to the caller."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidRequest(ServiceError):
    """The request is malformed or not allowed (a bad file type, a forbidden URL)."""


class PayloadTooLarge(ServiceError):
    """An upload exceeds its size limit."""


class QuotaExceeded(ServiceError):
    """The tenant is at a limit (stored chunks, daily pages, queued work)."""


class Unavailable(ServiceError):
    """A dependency the request needs (the job queue) is not available right now."""
