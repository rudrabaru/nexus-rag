"""Helpers and fixtures shared by the fetching tests."""
from unittest.mock import MagicMock

import httpx
import pytest

from src.crawling import readers, sitemap
from src.crawling.policy import DomainPacer
from src.jobs.contract import IngestionRequest


PAGE_TEXT = "word " * 40


ROBOTS_DENIAL = {
    "data": None, "code": 409, "name": "ResourcePolicyDenyError",
    "message": "Access to https://a.example/x is disallowed by site robots.txt: For User-Agent: *, Disallow: /x",
}


def jina_page(content=PAGE_TEXT, title="T", http_status=200):
    return httpx.Response(200, json={"code": 200, "data": {"title": title, "content": content, "httpStatus": http_status}})


JINA_SITEMAP = (
    "[https://a.example/docs/one](https://a.example/docs/one)  \n2024-01-01\n\n"
    "[https://a.example/blog/two](https://a.example/blog/two)  \n2024-01-01\n\n"
    "[https://a.example/img/logo.png](https://a.example/img/logo.png)\n"
)


class FakeApp:
    def __init__(self):
        self.deferred = []

    def configure_task(self, name, **options):
        app = self

        class _Deferrer:
            async def defer_async(self, **kwargs):
                app.deferred.append((name, options, kwargs))

        return _Deferrer()


def url_request(url="https://a.example/x"):
    return IngestionRequest(job_id="job-1", doc_id="doc-1", tenant_id="tenant-1", url=url)


@pytest.fixture
def web(monkeypatch):
    """Routes each reader's host to a queue of canned responses and records every request."""
    routes = {"r.jina.ai": [], "api.firecrawl.dev": []}
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        queue = routes[request.url.host]
        return queue.pop(0) if queue else httpx.Response(500, json={"message": "no canned response"})

    real_client = httpx.AsyncClient
    factory = lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)  # noqa: E731
    monkeypatch.setattr(readers.httpx, "AsyncClient", factory)
    monkeypatch.setattr(sitemap.httpx, "AsyncClient", factory)
    return routes, seen


@pytest.fixture
def fetch_env(monkeypatch):
    from src.jobs import fetch_tasks

    registry = MagicMock()
    registry.fetched_urls.return_value = set()
    registry.pages_fetched_today.return_value = 0
    monkeypatch.setattr(fetch_tasks, "FetchStore", lambda engine: registry)  # one mock stands in for both stores
    monkeypatch.setattr(fetch_tasks, "JobStore", lambda engine: registry)
    monkeypatch.setattr(fetch_tasks, "get_sync_engine", lambda: None)
    monkeypatch.setattr(fetch_tasks, "check_fetchable", lambda url, allowed, denied: None)
    monkeypatch.setattr(fetch_tasks, "_pacers", {0.0: DomainPacer(0.0)})
    monkeypatch.setenv("FETCH_MIN_INTERVAL_SECONDS", "0")
    from src.config import get_settings

    get_settings.cache_clear()
    return fetch_tasks, registry
