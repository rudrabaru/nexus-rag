"""
Fetching through reader APIs: robots.txt finality, reader fallback, sitemap discovery, and the
fetch task's quota and hand-off to the ingest queue. No real network: httpx.MockTransport.
"""
import json
from unittest.mock import MagicMock

import httpx
import pytest

from src.crawling import readers, sitemap
from src.crawling.policy import DomainPacer
from src.crawling.readers import ReaderError, RobotsBlockedError, read_page
from src.crawling.sitemap import discover_pages, filter_prefix, is_sitemap_url
from src.jobs.contract import FETCH_QUEUE, INGEST_QUEUE, INGEST_TASK, IngestionRequest

PAGE_TEXT = "word " * 40
ROBOTS_DENIAL = {
    "data": None, "code": 409, "name": "ResourcePolicyDenyError",
    "message": "Access to https://a.example/x is disallowed by site robots.txt: For User-Agent: *, Disallow: /x",
}


def jina_page(content=PAGE_TEXT, title="T", http_status=200):
    return httpx.Response(200, json={"code": 200, "data": {"title": title, "content": content, "httpStatus": http_status}})


def firecrawl_page(markdown=PAGE_TEXT):
    return httpx.Response(200, json={"success": True, "data": {"markdown": markdown, "metadata": {"title": "F", "statusCode": 200}}})


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


# ── Readers ──────────────────────────────────────────────────────────────────

async def test_jina_is_asked_to_honour_robots_txt_and_is_called_keyless(web):
    routes, seen = web
    routes["r.jina.ai"].append(jina_page())
    page = await read_page("https://a.example/x")
    assert page.provider == "jina" and page.title == "T"
    assert seen[0].headers["X-Robots-Txt"] and "authorization" not in seen[0].headers


async def test_a_robots_denial_is_final_and_never_routed_to_another_reader(web):
    routes, seen = web
    routes["r.jina.ai"].append(httpx.Response(409, json=ROBOTS_DENIAL))
    routes["api.firecrawl.dev"].append(firecrawl_page())
    with pytest.raises(RobotsBlockedError):
        await read_page("https://a.example/x", firecrawl_api_key="fc-key")
    assert [r.url.host for r in seen] == ["r.jina.ai"]


async def test_firecrawl_is_the_fallback_when_jina_cannot_read_the_page(web):
    routes, _ = web
    routes["r.jina.ai"].append(httpx.Response(429, json={"message": "rate limited"}))
    routes["api.firecrawl.dev"].append(firecrawl_page())
    page = await read_page("https://a.example/x", firecrawl_api_key="fc-key")
    assert page.provider == "firecrawl"


async def test_firecrawl_is_not_used_without_a_key(web):
    routes, seen = web
    routes["r.jina.ai"].append(httpx.Response(503, json={"message": "down"}))
    with pytest.raises(ReaderError):
        await read_page("https://a.example/x")
    assert [r.url.host for r in seen] == ["r.jina.ai"]


@pytest.mark.parametrize("response", [jina_page(content="  "), jina_page(http_status=404)])
async def test_an_empty_or_missing_page_is_an_error(web, response):
    routes, _ = web
    routes["r.jina.ai"].append(response)
    with pytest.raises(ReaderError):
        await read_page("https://a.example/x")


async def test_a_short_page_is_kept_because_short_is_not_evidence_of_uselessness(web):
    routes, _ = web
    routes["r.jina.ai"].append(jina_page(content="Access denied"))
    assert (await read_page("https://a.example/x")).markdown == "Access denied"


# ── Sitemaps ─────────────────────────────────────────────────────────────────

JINA_SITEMAP = (
    "[https://a.example/docs/one](https://a.example/docs/one)  \n2024-01-01\n\n"
    "[https://a.example/blog/two](https://a.example/blog/two)  \n2024-01-01\n\n"
    "[https://a.example/img/logo.png](https://a.example/img/logo.png)\n"
)


def test_sitemap_detection_and_filter_parsing():
    assert is_sitemap_url("https://a.example/sitemap.xml?filter=/docs/")
    assert is_sitemap_url("https://a.example/sitemap_index")
    assert not is_sitemap_url("https://a.example/docs/page")
    assert filter_prefix("https://a.example/sitemap.xml?filter=/Docs/") == "/docs/"


async def test_sitemap_pages_are_the_links_jina_renders_minus_media(web):
    routes, _ = web
    routes["r.jina.ai"].append(jina_page(content=JINA_SITEMAP))
    assert await discover_pages("https://a.example/sitemap.xml", max_pages=10) == [
        "https://a.example/docs/one", "https://a.example/blog/two",
    ]


async def test_sitemap_filter_and_cap_apply(web):
    routes, _ = web
    routes["r.jina.ai"].append(jina_page(content=JINA_SITEMAP))
    assert await discover_pages("https://a.example/sitemap.xml?filter=/blog/", max_pages=10) == ["https://a.example/blog/two"]
    routes["r.jina.ai"].append(jina_page(content=JINA_SITEMAP))
    assert len(await discover_pages("https://a.example/sitemap.xml", max_pages=1)) == 1


async def test_a_sitemap_index_is_read_one_level_deep(web):
    routes, seen = web
    index = "[https://a.example/sitemap-docs.xml](https://a.example/sitemap-docs.xml)\n" + PAGE_TEXT
    routes["r.jina.ai"] += [jina_page(content=index), jina_page(content=JINA_SITEMAP)]
    pages = await discover_pages("https://a.example/sitemap.xml", max_pages=10)
    assert "https://a.example/docs/one" in pages
    assert len(seen) == 2


# ── Per-domain pacing ────────────────────────────────────────────────────────

def test_pacing_spaces_requests_to_one_domain_but_not_across_domains():
    pacer = DomainPacer(min_interval=3.0)
    assert pacer.reserve("https://a.example/1") == 0.0
    assert pacer.reserve("https://a.example/2") == pytest.approx(3.0, abs=0.1)
    assert pacer.reserve("https://b.example/1") == 0.0


# ── The fetch task ───────────────────────────────────────────────────────────

class FakeApp:
    def __init__(self):
        self.deferred = []

    def configure_task(self, name, **options):
        app = self

        class _Deferrer:
            async def defer_async(self, **kwargs):
                app.deferred.append((name, options, kwargs))

        return _Deferrer()


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


def url_request(url="https://a.example/x"):
    return IngestionRequest(job_id="job-1", doc_id="doc-1", tenant_id="tenant-1", url=url)


async def test_fetched_pages_are_stored_audited_and_handed_to_the_ingest_queue(fetch_env, monkeypatch):
    fetch_tasks, registry = fetch_env

    async def fake_read(url, firecrawl_api_key=""):
        registry.fetched_urls.return_value = {url}
        return readers.FetchedPage(url=url, title="T", markdown=PAGE_TEXT, provider="jina")

    monkeypatch.setattr(fetch_tasks, "read_page", fake_read)
    app = FakeApp()
    await fetch_tasks._fetch(url_request(), app)

    registry.store_fetched_page.assert_called_once_with("job-1", "https://a.example/x", "T", PAGE_TEXT, "jina")
    assert registry.log_fetch.call_args.args[3] == "fetched"
    [(name, options, kwargs)] = app.deferred
    assert name == INGEST_TASK and options["queue"] == INGEST_QUEUE and options["lock"] == "doc-1"
    assert kwargs["job_id"] == "job-1"


async def test_a_robots_blocked_page_is_audited_and_fails_the_job_without_retrying(fetch_env, monkeypatch):
    fetch_tasks, registry = fetch_env

    async def blocked(url, firecrawl_api_key=""):
        raise RobotsBlockedError("disallowed by robots.txt")

    monkeypatch.setattr(fetch_tasks, "read_page", blocked)
    app = FakeApp()
    with pytest.raises(fetch_tasks.UnprocessableSourceError, match="robots.txt"):
        await fetch_tasks._fetch(url_request(), app)
    assert registry.log_fetch.call_args.args[3] == "robots_blocked"
    assert app.deferred == []


async def test_a_reader_outage_is_retried_by_the_queue(fetch_env, monkeypatch):
    fetch_tasks, _ = fetch_env

    async def down(url, firecrawl_api_key=""):
        raise ReaderError("jina HTTP 503")

    monkeypatch.setattr(fetch_tasks, "read_page", down)
    with pytest.raises(ReaderError):  # not UnprocessableSourceError: Procrastinate retries it
        await fetch_tasks._fetch(url_request(), FakeApp())


async def test_pages_beyond_the_daily_quota_are_skipped_not_fetched(fetch_env, monkeypatch):
    from src.config import get_settings

    fetch_tasks, registry = fetch_env
    registry.pages_fetched_today.return_value = get_settings().fetch_daily_page_quota
    monkeypatch.setattr(fetch_tasks, "read_page", MagicMock(side_effect=AssertionError("must not fetch")))
    with pytest.raises(fetch_tasks.UnprocessableSourceError, match="quota"):
        await fetch_tasks._fetch(url_request(), FakeApp())
    assert registry.log_fetch.call_args.args[3] == "quota_exceeded"


async def test_resuming_a_fully_indexed_document_completes_without_fetching(fetch_env, monkeypatch):
    fetch_tasks, registry = fetch_env
    monkeypatch.setattr(fetch_tasks, "_indexed_urls", lambda request: {"https://a.example/x"})
    monkeypatch.setattr(fetch_tasks, "read_page", MagicMock(side_effect=AssertionError("must not fetch")))
    app = FakeApp()

    await fetch_tasks._fetch(url_request().model_copy(update={"resume": True}), app)

    registry.update_job_status.assert_called_with("job-1", "complete", 100)
    assert app.deferred == []


def test_each_blueprint_only_holds_tasks_for_its_own_queue():
    """The parse worker (HF Space) must never be able to run a fetch task, and vice versa."""
    from src.jobs.fetch_tasks import fetch_blueprint
    from src.jobs.ingest_tasks import blueprint

    assert {t.queue for t in fetch_blueprint.tasks.values()} == {FETCH_QUEUE}
    assert {t.queue for t in blueprint.tasks.values()} == {INGEST_QUEUE}


def test_a_worker_app_refuses_tasks_from_another_queue():
    import procrastinate

    from src.jobs.workers import build_app

    stray = procrastinate.Blueprint()

    @stray.task(name="stray", queue=INGEST_QUEUE)
    def _stray():
        pass

    with pytest.raises(RuntimeError, match="must only run"):
        build_app(stray, FETCH_QUEUE)


def test_a_worker_app_with_only_its_own_tasks_builds():
    """
    Regression: Procrastinate's builtin remove_old_jobs task, which every App registers, was
    counted as foreign, so neither worker could start (found by a live run, not by this suite).
    """
    import procrastinate

    from src.jobs.workers import build_app

    own = procrastinate.Blueprint()

    @own.task(name="own", queue=FETCH_QUEUE)
    def _own():
        pass

    app = build_app(own, FETCH_QUEUE)
    assert any(name.startswith("builtin:") or "builtin_tasks" in name for name in app.tasks)


def test_jina_robots_denial_payload_matches_what_the_reader_detects():
    """Guards the parsing of the real 409 body observed on 2026-09-26."""
    assert "robots" in json.dumps(ROBOTS_DENIAL["message"]).lower()


# ── Policy gaps: sitemap detection, cross-host links, child sitemaps, redirects, redaction ──

@pytest.mark.parametrize("url", [
    "https://a.example/sitemap.xml", "https://a.example/wp-sitemap.xml", "https://a.example/sitemap_index",
    "https://a.example/sitemap.xml.gz?filter=/docs/",
])
def test_sitemap_files_are_recognised(url):
    assert is_sitemap_url(url)


@pytest.mark.parametrize("url", [
    "https://a.example/blog/sitemap-guide", "https://a.example/feed.xml", "https://a.example/docs/page",
    "https://a.example/sitemaps/overview",
])
def test_pages_that_merely_mention_sitemaps_are_ordinary_pages(url):
    assert not is_sitemap_url(url)


async def test_a_sitemap_cannot_aim_fetches_at_another_host(web):
    routes, _ = web
    hostile = "[one](https://a.example/docs/one)\n[evil](https://evil.example/admin)\n[sub](https://docs.a.example/x)\n" + PAGE_TEXT
    routes["r.jina.ai"].append(jina_page(content=hostile))
    pages = await discover_pages("https://a.example/sitemap.xml", max_pages=10)
    assert pages == ["https://a.example/docs/one", "https://docs.a.example/x"]


async def test_a_refused_child_sitemap_is_never_read(web):
    from src.ingestion.url_policy import UnsafeUrlError

    routes, seen = web
    index = "[c](https://a.example/sitemap-docs.xml)\n" + PAGE_TEXT
    routes["r.jina.ai"] += [jina_page(content=index), jina_page(content=JINA_SITEMAP)]
    asked = []

    async def refuse(child):
        asked.append(child)
        raise UnsafeUrlError("denied")

    await discover_pages("https://a.example/sitemap.xml", max_pages=10, authorize=refuse)
    assert asked == ["https://a.example/sitemap-docs.xml"]
    assert len(seen) == 1  # only the index itself was read


async def test_a_redirect_to_a_forbidden_host_is_denied_not_stored(fetch_env, monkeypatch):
    from src.ingestion.url_policy import UnsafeUrlError

    fetch_tasks, registry = fetch_env

    def policy(url, allowed, denied):
        if "evil.example" in url:
            raise UnsafeUrlError("denied host")

    monkeypatch.setattr(fetch_tasks, "check_fetchable", policy)

    async def redirected(url, firecrawl_api_key=""):
        return readers.FetchedPage(url=url, title="T", markdown=PAGE_TEXT, provider="jina", final_url="https://evil.example/landing")

    monkeypatch.setattr(fetch_tasks, "read_page", redirected)
    with pytest.raises(fetch_tasks.UnprocessableSourceError):
        await fetch_tasks._fetch(url_request(), FakeApp())
    registry.store_fetched_page.assert_not_called()
    assert registry.log_fetch.call_args.args[3] == "denied"


async def test_audit_rows_never_carry_query_strings(fetch_env, monkeypatch):
    fetch_tasks, registry = fetch_env

    async def fake_read(url, firecrawl_api_key=""):
        registry.fetched_urls.return_value = {url}
        return readers.FetchedPage(url=url, title="T", markdown=PAGE_TEXT, provider="jina")

    monkeypatch.setattr(fetch_tasks, "read_page", fake_read)
    await fetch_tasks._fetch(url_request("https://a.example/x?token=SECRET#frag"), FakeApp())
    assert all("SECRET" not in str(call.args) for call in registry.log_fetch.call_args_list)
