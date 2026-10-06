"""What the fetch worker may ask a reader API for."""
import pytest
from src.crawling.policy import check_fetchable
from src.crawling.url_policy import UnsafeUrlError
from src.services.errors import InvalidRequest, QuotaExceeded
from tests.support.ingest_security import Wired, resolve_to


NO_DOMAIN_LISTS = ([], [])


def test_plain_http_is_not_fetched(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    with pytest.raises(UnsafeUrlError):
        check_fetchable("http://example.com/docs", *NO_DOMAIN_LISTS)


@pytest.mark.parametrize("url", ["https://facebook.com/someone", "https://m.facebook.com/someone"])
def test_denied_domains_and_their_subdomains_are_not_fetched(url, monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    with pytest.raises(UnsafeUrlError):
        check_fetchable(url, [], ["facebook.com"])


def test_a_domain_that_merely_ends_with_a_denied_name_is_allowed(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    check_fetchable("https://notfacebook.com/page", [], ["facebook.com"])


def test_allowlist_mode_rejects_everything_else(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    check_fetchable("https://docs.python.org/3/", ["docs.python.org"], [])
    with pytest.raises(UnsafeUrlError):
        check_fetchable("https://example.com/", ["docs.python.org"], [])


def test_the_fetch_policy_still_blocks_private_addresses(monkeypatch):
    resolve_to(monkeypatch, "10.0.0.5")
    with pytest.raises(UnsafeUrlError):
        check_fetchable("https://internal.example/", *NO_DOMAIN_LISTS)


@pytest.mark.asyncio
async def test_http_urls_are_rejected_by_the_api_before_any_job_exists(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    wired = Wired()
    with pytest.raises(InvalidRequest):
        await wired.submit(url="http://example.com/docs")
    wired.jobs.register_job.assert_not_called()


@pytest.mark.asyncio
async def test_a_tenant_over_its_daily_page_quota_gets_429(monkeypatch):
    from src.config import get_settings

    resolve_to(monkeypatch, "93.184.216.34")
    wired = Wired(pages_fetched_today=get_settings().fetch_daily_page_quota)
    with pytest.raises(QuotaExceeded):
        await wired.submit(url="https://example.com/docs")
    wired.jobs.register_job.assert_not_called()
