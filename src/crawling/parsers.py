import re
from urllib.parse import urljoin, urlparse
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def extract_title(result) -> str:
    """Extract page title from crawl result."""
    # Try metadata first (most reliable)
    if (
        hasattr(result, "metadata")
        and result.metadata
        and isinstance(result.metadata, dict)
    ):
        if "title" in result.metadata:
            return result.metadata["title"]

    # Fallback: extract from markdown (first heading)
    if hasattr(result, "markdown") and result.markdown:
        lines = result.markdown.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line.replace("# ", "").strip()

    # Last resort: use URL
    return urlparse(result.url if hasattr(result, "url") else "").path.split("/")[-1]


def extract_links(
    base_url: str, markdown: str, crawl_links: Optional[dict] = None
) -> List[str]:
    """
    Extract links from markdown or Crawl4AI result.

    Args:
        base_url: Base URL for resolving relative links
        markdown: Markdown content to extract links from
        crawl_links: Links dict from CrawlResult with 'internal' and 'external' keys

    Returns:
        List of absolute URLs
    """
    links_set = set()

    try:
        if crawl_links and isinstance(crawl_links, dict):
            internal_links = crawl_links.get("internal", [])
            if isinstance(internal_links, list):
                for link in internal_links:
                    if isinstance(link, str) and link.strip():
                        links_set.add(link.strip())

            external_links = crawl_links.get("external", [])
            if isinstance(external_links, list):
                for link in external_links:
                    if isinstance(link, str) and link.strip():
                        links_set.add(link.strip())
    except Exception as e:
        logger.debug(f"Error extracting Crawl4AI links: {e}")

    # Also extract links from markdown (markdown syntax [text](url))
    try:
        markdown_links = re.findall(r"\[.+?\]\((.+?)\)", markdown)
        for link in markdown_links:
            if link and isinstance(link, str):
                abs_url = urljoin(base_url, link.strip())
                links_set.add(abs_url)
    except Exception as e:
        logger.debug(f"Error extracting markdown links: {e}")

    # Convert to sorted list for consistency
    return sorted(list(links_set))
