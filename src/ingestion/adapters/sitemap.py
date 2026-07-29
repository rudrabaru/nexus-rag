import logging
import xml.etree.ElementTree as ET
from typing import List, Set
from urllib.parse import urljoin, urlparse, parse_qs
import httpx

from src.crawling.metadata import CrawledDocument, AdapterResult
from src.ingestion.base import IngestionAdapter

logger = logging.getLogger(__name__)

IGNORED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".xml", ".json", ".csv", ".xlsx", ".docx",
    ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2", ".ttf", ".eot"
}

MAX_SITEMAP_PAGES = 500

class SitemapAdapter(IngestionAdapter):
    """
    Adapter for parsing XML sitemaps and extracting clean canonical URLs.
    Does not perform deep crawling; simply extracts and filters page URLs from sitemap.xml.
    Supports 1 level of recursion for <sitemapindex> files (up to 10 child sitemaps).
    """

    async def _fetch_xml(self, client: httpx.AsyncClient, url: str) -> bytes:
        logger.info(f"SitemapAdapter fetching XML: {url}")
        
        import socket
        import ipaddress
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if hostname:
                ip = socket.gethostbyname(hostname)
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                    raise ValueError(f"URL resolves to private/internal IP: {ip}")
        except Exception as e:
            logger.error(f"SSRF validation failed for sitemap {url}: {e}")
            raise ValueError(f"Invalid or restricted URL: {e}")

        resp = await client.get(url, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
        return resp.content

    def _extract_urls_from_xml(self, xml_content: bytes) -> tuple[List[str], List[str]]:
        page_urls: List[str] = []
        child_sitemaps: List[str] = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parse error in sitemap: {e}")
            return page_urls, child_sitemaps

        namespaces = {}
        if root.tag.startswith("{"):
            ns_uri = root.tag.split("}")[0].strip("{")
            namespaces["ns"] = ns_uri

        tag_prefix = "ns:" if namespaces else ""

        if "sitemapindex" in root.tag.lower():
            for sitemap_elem in root.findall(f".//{tag_prefix}sitemap", namespaces):
                loc_elem = sitemap_elem.find(f"{tag_prefix}loc", namespaces)
                if loc_elem is not None and loc_elem.text:
                    child_sitemaps.append(loc_elem.text.strip())
        else:
            for url_elem in root.findall(f".//{tag_prefix}url", namespaces):
                loc_elem = url_elem.find(f"{tag_prefix}loc", namespaces)
                if loc_elem is not None and loc_elem.text:
                    page_urls.append(loc_elem.text.strip())

        if not page_urls and not child_sitemaps:
            for elem in root.iter():
                if elem.tag.endswith("loc") and elem.text:
                    text_url = elem.text.strip()
                    if text_url.lower().endswith(".xml") or "sitemap" in text_url.lower():
                        child_sitemaps.append(text_url)
                    else:
                        page_urls.append(text_url)

        return page_urls, child_sitemaps

    def _is_valid_page_url(self, url: str) -> bool:
        url_lower = url.lower().split("?")[0].split("#")[0]
        if any(url_lower.endswith(ext) for ext in IGNORED_EXTENSIONS):
            return False
        return True

    async def ingest(self, source: str, extract_visuals: bool = False, **kwargs) -> AdapterResult:
        logger.info(f"SitemapAdapter processing sitemap: {source}")
        
        filter_prefix = kwargs.get("filter_prefix") or kwargs.get("filter")
        fetch_url = source
        if "?" in source:
            parsed = urlparse(source)
            qs = parse_qs(parsed.query)
            if "filter" in qs and qs["filter"]:
                filter_prefix = qs["filter"][0]
            fetch_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        seen_urls: Set[str] = set()
        clean_urls: List[str] = []

        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (NexusRAG/1.0)"}) as client:
            try:
                xml_content = await self._fetch_xml(client, fetch_url)
            except Exception as e:
                logger.error(f"Failed to fetch sitemap {fetch_url}: {e}")
                raise ValueError(f"Failed to fetch sitemap URL: {e}")

            page_urls, child_sitemaps = self._extract_urls_from_xml(xml_content)

            # Process top-level URLs
            for u in page_urls:
                if len(clean_urls) >= MAX_SITEMAP_PAGES:
                    logger.warning(f"Sitemap reached maximum capacity of {MAX_SITEMAP_PAGES} pages.")
                    break
                if u not in seen_urls and self._is_valid_page_url(u):
                    seen_urls.add(u)
                    clean_urls.append(u)

            # Handle sitemapindex recursion (max 10 child sitemaps, 1 level deep)
            if child_sitemaps and len(clean_urls) < MAX_SITEMAP_PAGES:
                logger.info(f"Sitemap index detected with {len(child_sitemaps)} child sitemaps. Recursively fetching up to 10...")
                for child_url in child_sitemaps[:10]:
                    if len(clean_urls) >= MAX_SITEMAP_PAGES:
                        break
                    try:
                        child_xml = await self._fetch_xml(client, child_url)
                        child_pages, _ = self._extract_urls_from_xml(child_xml)
                        for u in child_pages:
                            if len(clean_urls) >= MAX_SITEMAP_PAGES:
                                logger.warning(f"Sitemap reached maximum capacity of {MAX_SITEMAP_PAGES} pages during recursion.")
                                break
                            if u not in seen_urls and self._is_valid_page_url(u):
                                seen_urls.add(u)
                                clean_urls.append(u)
                    except Exception as child_err:
                        logger.warning(f"Failed to fetch child sitemap {child_url}: {child_err}")

        if filter_prefix and str(filter_prefix).strip():
            pref = str(filter_prefix).strip().lower()
            clean_urls = [u for u in clean_urls if pref in u.lower()]
            logger.info(f"SitemapAdapter filtered URLs by prefix '{filter_prefix}': {len(clean_urls)} URLs remain")

        if not clean_urls and not child_sitemaps:
            logger.warning(f"No <loc> URLs found in {source}. Might be an RSS feed or empty sitemap.")
            # Fallback will be handled in dispatcher or ingestion service if 0 docs returned
            return AdapterResult(documents=[], visual_chunks=[])

        logger.info(f"SitemapAdapter successfully extracted {len(clean_urls)} clean URLs from {source}")

        # Return CrawledDocument stubs where url is set, but markdown_content is empty (to be fetched in ingestion_service)
        docs = [CrawledDocument(url=u, title=f"Sitemap Page: {u}", markdown_content="") for u in clean_urls]
        return AdapterResult(documents=docs, visual_chunks=[])
