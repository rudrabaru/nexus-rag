"""Shared by the integration tests."""
from sqlalchemy.engine import Engine

from src.stores.documents import DocumentStore
from src.stores.fetches import FetchStore
from src.stores.jobs import JobStore


class Stores:
    """Test convenience: the document, job and fetch stores behind one name. Production code takes the one it needs."""

    def __init__(self, engine: Engine):
        self._stores = (DocumentStore(engine), JobStore(engine), FetchStore(engine))

    def __getattr__(self, name):
        for store in self._stores:
            if hasattr(store, name):
                return getattr(store, name)
        raise AttributeError(name)
