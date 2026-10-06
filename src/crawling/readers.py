"""
Hosted reader APIs: the only way the platform reads a web page.

No process we host sends a request to a third-party website. A reader fetches the page from
its own network, under its own terms, and returns Markdown. Readers are tried in order:

1. Jina Reader, keyless (verified 2026-09-26): ~20 requests/min per IP, and it does not draw
   on Jina's one-time token grant. `X-Robots-Txt` makes it check the site's robots.txt first
   and answer HTTP 409 ResourcePolicyDenyError when the page is disallowed. Also reads PDFs
   by URL, so a remote PDF is never downloaded by us.
2. Firecrawl (only when FIRECRAWL_API_KEY is set): 1,000 free credits/month. Its scrape docs
   do not state robots.txt handling, which is why it is second.

A robots.txt denial is final: the page is recorded as robots_blocked and never retried with
another reader, since that would be routing around the site owner's decision.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import httpx

from src.crawling.url_policy import redact_url

logger = logging.getLogger(__name__)

USER_AGENT = "NexusRAG"  # the robots.txt user agent readers check on our behalf
READ_TIMEOUT_SECONDS = 60.0


class RobotsBlockedError(Exception):
    """The site's robots.txt disallows the page. Final: never retried or routed around."""


class ReaderError(Exception):
    """A reader could not produce the page. The next reader may succeed."""


@dataclass
class FetchedPage:
    url: str
    title: Optional[str]
    markdown: str
    provider: str
    final_url: Optional[str] = None  # where the reader ended up after redirects, when it says so


async def read_with_jina(client: httpx.AsyncClient, url: str) -> FetchedPage:
    response = await client.get(
        f"https://r.jina.ai/{url}",
        headers={
            "Accept": "application/json",
            "X-Return-Format": "markdown",
            "X-Retain-Images": "none",
            "X-Robots-Txt": USER_AGENT,
            "X-Timeout": "30",
        },
    )
    body = _json(response)
    if response.status_code == 409 and "robots" in str(body.get("message", "")).lower():
        raise RobotsBlockedError(body.get("message"))
    if response.status_code >= 400:
        raise ReaderError(f"jina HTTP {response.status_code}: {str(body.get('message') or response.text)[:160]}")
    data = body.get("data") or {}
    if isinstance(data.get("httpStatus"), int) and data["httpStatus"] >= 400:
        raise ReaderError(f"jina: the site answered HTTP {data['httpStatus']}")
    return FetchedPage(
        url=url, title=data.get("title"), markdown=data.get("content") or "", provider="jina", final_url=data.get("url")
    )


async def read_with_firecrawl(client: httpx.AsyncClient, url: str, api_key: str) -> FetchedPage:
    response = await client.post(
        "https://api.firecrawl.dev/v2/scrape",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
    )
    body = _json(response)
    if response.status_code >= 400 or not body.get("success"):
        raise ReaderError(f"firecrawl HTTP {response.status_code}: {str(body.get('error') or response.text)[:160]}")
    data = body.get("data") or {}
    metadata = data.get("metadata") or {}
    if isinstance(metadata.get("statusCode"), int) and metadata["statusCode"] >= 400:
        raise ReaderError(f"firecrawl: the site answered HTTP {metadata['statusCode']}")
    title = metadata.get("title")
    return FetchedPage(
        url=url, title=title[0] if isinstance(title, list) and title else title,
        markdown=data.get("markdown") or "", provider="firecrawl",
        final_url=metadata.get("url") or metadata.get("sourceURL"),
    )


async def read_page(url: str, firecrawl_api_key: str = "") -> FetchedPage:
    """The page as Markdown from the first reader that succeeds. Raises RobotsBlockedError or ReaderError."""
    errors: List[str] = []
    async with httpx.AsyncClient(timeout=READ_TIMEOUT_SECONDS) as client:
        readers = [lambda: read_with_jina(client, url)]
        if firecrawl_api_key:
            readers.append(lambda: read_with_firecrawl(client, url, firecrawl_api_key))
        for read in readers:
            try:
                return _readable(await read())
            except RobotsBlockedError:
                raise
            except (ReaderError, httpx.HTTPError) as e:
                errors.append(str(e) or type(e).__name__)
                logger.warning(f"FETCH | {redact_url(url)} | {errors[-1]}")
    raise ReaderError("; ".join(errors))


def _readable(page: FetchedPage) -> FetchedPage:
    """
    A page with no words at all is an empty shell and the next reader is tried. A short page is not
    rejected for being short: length is not evidence that content is useless, and a login wall or bot
    block that comes back for several URLs is recognised by being identical across them (the fetch
    job skips repeats of the same content).
    """
    if not page.markdown.split():
        raise ReaderError(f"{page.provider}: no text extracted (empty page)")
    return page


def _json(response: httpx.Response) -> dict:
    try:
        body = response.json()
        return body if isinstance(body, dict) else {}
    except ValueError:
        return {}
