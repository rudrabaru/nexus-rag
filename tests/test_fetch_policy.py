"""Fetch-policy gaps: sitemap detection, cross-host links, child sitemaps."""
import pytest
from src.crawling import readers
from src.crawling.sitemap import discover_pages, is_sitemap_url
from tests.support.fetching import FakeApp, JINA_SITEMAP, PAGE_TEXT, jina_page, url_request


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
    from src.crawling.url_policy import UnsafeUrlError

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
    from src.crawling.url_policy import UnsafeUrlError

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
