"""The fetch task: what it stores, skips and reports."""
import json
from unittest.mock import MagicMock
import pytest
from src.crawling import readers
from src.crawling.readers import ReaderError, RobotsBlockedError
from src.jobs.contract import FETCH_QUEUE, INGEST_QUEUE, INGEST_TASK
from tests.support.fetching import FakeApp, PAGE_TEXT, ROBOTS_DENIAL, url_request


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
