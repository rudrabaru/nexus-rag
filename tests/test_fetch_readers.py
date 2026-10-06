"""Readers, sitemaps and per-domain pacing."""
import httpx
import pytest
from src.crawling.policy import DomainPacer
from src.crawling.readers import ReaderError, RobotsBlockedError, read_page
from src.crawling.sitemap import discover_pages, filter_prefix, is_sitemap_url
from tests.support.fetching import JINA_SITEMAP, PAGE_TEXT, ROBOTS_DENIAL, jina_page


def firecrawl_page(markdown=PAGE_TEXT):
    return httpx.Response(200, json={"success": True, "data": {"markdown": markdown, "metadata": {"title": "F", "statusCode": 200}}})


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


def test_pacing_spaces_requests_to_one_domain_but_not_across_domains():
    pacer = DomainPacer(min_interval=3.0)
    assert pacer.reserve("https://a.example/1") == 0.0
    assert pacer.reserve("https://a.example/2") == pytest.approx(3.0, abs=0.1)
    assert pacer.reserve("https://b.example/1") == 0.0
