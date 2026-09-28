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
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from src.crawling.readers import READ_TIMEOUT_SECONDS, ReaderError, RobotsBlockedError, read_with_jina

logger = logging.getLogger(__name__)

MAX_CHILD_SITEMAPS = 10
# Media and archives are not documents a reader can turn into text.
_NON_DOCUMENT_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".zip", ".tar", ".gz", ".json", ".csv",
    ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2", ".ttf", ".eot",
)
_URL = re.compile(r"https?://[^\s<>()\[\]\"']+")


def is_sitemap_url(url: str) -> bool:
    """The rule the API has always used to decide a URL is a sitemap."""
    lowered = url.lower().split("?")[0]
    return lowered.endswith(".xml") or "sitemap" in lowered


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


async def discover_pages(url: str, max_pages: int, firecrawl_api_key: str = "") -> List[str]:
    """Page URLs listed by the sitemap at `url` (one index level deep), filtered and capped."""
    prefix = filter_prefix(url)
    async with httpx.AsyncClient(timeout=READ_TIMEOUT_SECONDS) as client:
        try:
            entries = await _read_sitemap(client, url)
            children = [e for e in entries if _path(e).endswith(".xml")][:MAX_CHILD_SITEMAPS]
            pages = [e for e in entries if not _path(e).endswith(".xml")]
            for child in children:
                if len(pages) >= max_pages * 4:  # enough candidates to fill the cap after filtering
                    break
                try:
                    pages.extend(e for e in await _read_sitemap(client, child) if not _path(e).endswith(".xml"))
                except (ReaderError, RobotsBlockedError, httpx.HTTPError) as e:
                    logger.warning(f"SITEMAP | child sitemap {child} skipped: {e}")
        except (ReaderError, httpx.HTTPError) as e:
            if not firecrawl_api_key:
                raise
            logger.warning(f"SITEMAP | jina could not read {url} ({e}); trying firecrawl map")
            pages = await _map_with_firecrawl(client, url, firecrawl_api_key, limit=max_pages * 4)

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
    logger.info(f"SITEMAP | {url} | {len(pages)} listed, {len(selected)} selected (filter={prefix!r}, cap={max_pages})")
    return selected
