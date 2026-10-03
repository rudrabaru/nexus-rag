"""
Sitemap discovery through reader APIs: the sitemap XML is never downloaded by us.

Jina Reader renders a sitemap as a list of its <loc> URLs (verified 2026-09-26), so the page
URLs are the absolute links in that output. A sitemap index lists child sitemaps, which are
themselves XML files; those are read one level deep. With FIRECRAWL_API_KEY set, Firecrawl's
map endpoint (sitemap-only mode) is the fallback.

Only an explicit sitemap URL triggers multi-page ingestion. A plain page URL is one page: the
previous dispatcher silently expanded any page into its whole site via robots.txt, which
spent fetch quota and site traffic the user never asked for.
"""
import logging
import re
from typing import Awaitable, Callable, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from src.crawling.readers import READ_TIMEOUT_SECONDS, ReaderError, RobotsBlockedError, read_with_jina
from src.ingestion.url_policy import UnsafeUrlError, redact_url

logger = logging.getLogger(__name__)

MAX_CHILD_SITEMAPS = 10
MAX_SITEMAP_PAGES = 50  # bounds one job's reader calls, fetch quota and memory
# Media and archives are not documents a reader can turn into text.
_NON_DOCUMENT_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".zip", ".tar", ".gz", ".json", ".csv",
    ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2", ".ttf", ".eot",
)
_URL = re.compile(r"https?://[^\s<>()\[\]\"']+")


_SITEMAP_NAMES = {"sitemap", "sitemap_index", "sitemap-index"}


def is_sitemap_url(url: str) -> bool:
    """
    Whether the URL names a sitemap file: an XML file with "sitemap" in its name, or a bare
    sitemap name. "sitemap" elsewhere in the path (an article about sitemaps) and other XML
    (feeds) are ordinary pages: treating them as sitemaps would fan one page out into a crawl.
    """
    name = urlparse(url.strip()).path.lower().rsplit("/", 1)[-1]
    return name in _SITEMAP_NAMES or (name.endswith((".xml", ".xml.gz")) and "sitemap" in name)


def _same_site(link: str, sitemap_url: str) -> bool:
    """A sitemap may list its own site (and its subdomains), not other hosts: a hostile sitemap could otherwise aim our fetches anywhere."""
    host, own = (urlparse(link).hostname or "").lower(), (urlparse(sitemap_url).hostname or "").lower()
    return bool(host) and (host == own or host.endswith("." + own) or own.endswith("." + host))


def filter_prefix(url: str) -> Optional[str]:
    """`?filter=/docs/` on a sitemap URL keeps only page URLs containing that text."""
    values = parse_qs(urlparse(url).query).get("filter")
    return values[0].strip().lower() if values and values[0].strip() else None


def _path(url: str) -> str:
    return urlparse(url).path.lower()


def _links(markdown: str, own_url: str) -> List[str]:
    seen, links = {own_url.split("?")[0]}, []
    for match in _URL.findall(markdown):
        link = match.rstrip(".,;")
        if link not in seen:
            seen.add(link)
            links.append(link)
    return links


async def _read_sitemap(client: httpx.AsyncClient, url: str) -> List[str]:
    page = await read_with_jina(client, url.split("?")[0])
    return _links(page.markdown, url)


async def _map_with_firecrawl(client: httpx.AsyncClient, url: str, api_key: str, limit: int) -> List[str]:
    parsed = urlparse(url)
    response = await client.post(
        "https://api.firecrawl.dev/v2/map",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"url": f"{parsed.scheme}://{parsed.netloc}", "sitemap": "only", "limit": limit},
    )
    if response.status_code >= 400:
        raise ReaderError(f"firecrawl map HTTP {response.status_code}: {response.text[:160]}")
    return [link["url"] if isinstance(link, dict) else link for link in response.json().get("links", [])]


async def discover_pages(
    url: str, max_pages: int, firecrawl_api_key: str = "",
    authorize: Optional[Callable[[str], Awaitable[None]]] = None,
) -> List[str]:
    """
    Page URLs listed by the sitemap at `url` (one index level deep), filtered and capped.
    `authorize` runs before every child sitemap is read (policy, pacing, audit); a child it refuses
    with UnsafeUrlError is skipped.
    """
    prefix = filter_prefix(url)
    async with httpx.AsyncClient(timeout=READ_TIMEOUT_SECONDS) as client:
        try:
            entries = [e for e in await _read_sitemap(client, url) if _same_site(e, url)]
            children = [e for e in entries if _path(e).endswith(".xml")][:MAX_CHILD_SITEMAPS]
            pages = [e for e in entries if not _path(e).endswith(".xml")]
            for child in children:
                if len(pages) >= max_pages * 4:  # enough candidates to fill the cap after filtering
                    break
                try:
                    if authorize:
                        await authorize(child)
                    listed = await _read_sitemap(client, child)
                    pages.extend(e for e in listed if _same_site(e, url) and not _path(e).endswith(".xml"))
                except (ReaderError, RobotsBlockedError, UnsafeUrlError, httpx.HTTPError) as e:
                    logger.warning(f"SITEMAP | child sitemap {redact_url(child)} skipped: {e}")
        except (ReaderError, httpx.HTTPError) as e:
            if not firecrawl_api_key:
                raise
            logger.warning(f"SITEMAP | jina could not read {redact_url(url)} ({e}); trying firecrawl map")
            pages = [p for p in await _map_with_firecrawl(client, url, firecrawl_api_key, limit=max_pages * 4) if _same_site(p, url)]

    selected, seen = [], set()
    for page in pages:
        if page in seen or _path(page).endswith(_NON_DOCUMENT_EXTENSIONS):
            continue
        if prefix and prefix not in page.lower():
            continue
        seen.add(page)
        selected.append(page)
        if len(selected) >= max_pages:
            break
    logger.info(f"SITEMAP | {redact_url(url)} | {len(pages)} listed, {len(selected)} selected (filter={prefix!r}, cap={max_pages})")
    return selected
