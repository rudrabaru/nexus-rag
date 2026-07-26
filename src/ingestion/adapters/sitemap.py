import logging
import xml.etree.ElementTree as ET
from typing import List, Set
from urllib.parse import urljoin
import httpx

from src.crawling.metadata import CrawledDocument, AdapterResult
from src.ingestion.base import IngestionAdapter

logger = logging.getLogger(__name__)

IGNORED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".xml", ".json", ".csv", ".xlsx", ".docx",
    ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2", ".ttf", ".eot"
}

class SitemapAdapter(IngestionAdapter):
    """
    Adapter for parsing XML sitemaps and extracting clean canonical URLs.
    Does not perform deep crawling; simply extracts and filters page URLs from sitemap.xml.
    Supports 1 level of recursion for <sitemapindex> files (up to 10 child sitemaps).
    """

    async def _fetch_xml(self, client: httpx.AsyncClient, url: str) -> bytes:
        logger.info(f"SitemapAdapter fetching XML: {url}")
        resp = await client.get(url, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
        return resp.content

    def _extract_urls_from_xml(self, xml_content: bytes) -> tuple[List[str], List[str]]:
        """
        Parses XML content and returns (page_urls, child_sitemap_urls).
        Handles standard XML namespaces automatically by ignoring them or stripping them.
        """
        page_urls = []
        child_sitemaps = []
        try:
            root = ET.fromstring(xml_content)
        except Exception as e:
            logger.warning(f"Failed to parse XML sitemap: {e}")
            return [], []

        # Iterate over all elements regardless of xmlns tag prefix
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == "url":
                loc = None
                for child in elem:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag == "loc" and child.text:
                        loc = child.text.strip()
                        break
                if loc:
                    page_urls.append(loc)
            elif tag == "sitemap":
                loc = None
                for child in elem:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag == "loc" and child.text:
                        loc = child.text.strip()
                        break
                if loc:
                    child_sitemaps.append(loc)

        return page_urls, child_sitemaps

    def _is_valid_page_url(self, url: str) -> bool:
        url_lower = url.lower().split("?")[0].split("#")[0]
        if any(url_lower.endswith(ext) for ext in IGNORED_EXTENSIONS):
            return False
        return True

    async def ingest(self, source: str, extract_visuals: bool = False, **kwargs) -> AdapterResult:
        logger.info(f"SitemapAdapter processing sitemap: {source}")
        
        seen_urls: Set[str] = set()
        clean_urls: List[str] = []

        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (NexusRAG/1.0)"}) as client:
            try:
                xml_content = await self._fetch_xml(client, source)
            except Exception as e:
                logger.error(f"Failed to fetch sitemap {source}: {e}")
                raise ValueError(f"Failed to fetch sitemap URL: {e}")

            page_urls, child_sitemaps = self._extract_urls_from_xml(xml_content)

            # Process top-level URLs
            for u in page_urls:
                if u not in seen_urls and self._is_valid_page_url(u):
                    seen_urls.add(u)
                    clean_urls.append(u)

            # Handle sitemapindex recursion (max 10 child sitemaps, 1 level deep)
            if child_sitemaps:
                logger.info(f"Sitemap index detected with {len(child_sitemaps)} child sitemaps. Recursively fetching up to 10...")
                for child_url in child_sitemaps[:10]:
                    try:
                        child_xml = await self._fetch_xml(client, child_url)
                        child_pages, _ = self._extract_urls_from_xml(child_xml)
                        for u in child_pages:
                            if u not in seen_urls and self._is_valid_page_url(u):
                                seen_urls.add(u)
                                clean_urls.append(u)
                    except Exception as child_err:
                        logger.warning(f"Failed to fetch child sitemap {child_url}: {child_err}")

        if not clean_urls and not child_sitemaps:
            logger.warning(f"No <loc> URLs found in {source}. Might be an RSS feed or empty sitemap.")
            # Fallback will be handled in dispatcher or ingestion service if 0 docs returned
            return AdapterResult(documents=[], visual_chunks=[])

        logger.info(f"SitemapAdapter successfully extracted {len(clean_urls)} clean URLs from {source}")

        # Return CrawledDocument stubs where url is set, but markdown_content is empty (to be fetched in ingestion_service)
        docs = [CrawledDocument(url=u, title=f"Sitemap Page: {u}", markdown_content="") for u in clean_urls]
        return AdapterResult(documents=docs, visual_chunks=[])
