class UnprocessableSourceError(ValueError):
    """The source cannot be ingested as it is (no content, too large, unreadable). Retrying cannot help."""
