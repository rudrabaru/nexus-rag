"""Failures that several layers need to agree on."""


class UnprocessableSourceError(ValueError):
    """
    A source that cannot be turned into chunks, or work that cannot succeed: retrying would give
    the same result (no usable content, content too large, a rejected credential). The job is
    failed with this message instead of being retried. The message is safe to show to the owner.
    """


class EmbeddingRejectedError(UnprocessableSourceError):
    """The embedding provider refused a request in a way retrying cannot fix: a bad key, a wrong model, a wrong vector width."""
