import logging
from src.crawling.metadata import AdapterResult
from src.ingestion.adapters.web import WebAdapter
from src.ingestion.adapters.universal_adapter import UniversalAdapter
from src.ingestion.adapters.sitemap import SitemapAdapter

logger = logging.getLogger(__name__)


class IngestionDispatcher:
    """
    Detects the input format and routes it to the appropriate adapter.
    """

    def __init__(self):
        self.web_adapter = WebAdapter()
        self.universal_adapter = UniversalAdapter()
        self.sitemap_adapter = SitemapAdapter()

    async def _detect_http_content_type(self, url: str) -> str:
        """HEAD request to detect actual content type before routing."""
        try:
            import httpx

            async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
                resp = await client.head(url)
                ct = resp.headers.get("content-type", "").lower()
                if "pdf" in ct:
                    return "pdf"
                if "text/plain" in ct or url.lower().endswith(".txt"):
                    return "txt"
                if url.lower().endswith(".md"):
                    return "md"
        except Exception:
            pass
        return "html"

    async def _discover_sitemap(self, url: str) -> str | None:
        """
        Attempts to automatically discover the sitemap for a given URL.
        Returns the sitemap URL if found, otherwise None.
        """
        import httpx
        from urllib.parse import urlparse, urljoin

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
                # 1. Check robots.txt for Sitemap directive
                robots_url = urljoin(base_url, "/robots.txt")
                try:
                    resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        for line in resp.text.splitlines():
                            if line.lower().startswith("sitemap:"):
                                sitemap_url = line.split(":", 1)[1].strip()
                                logger.info(f"DISPATCHER | Auto-discovered sitemap via robots.txt: {sitemap_url}")
                                return sitemap_url
                except Exception:
                    pass

                # 2. Check common sitemap paths
                common_paths = ["/sitemap.xml", "/sitemap_index.xml"]
                for path in common_paths:
                    sitemap_url = urljoin(base_url, path)
                    try:
                        resp = await client.head(sitemap_url)
                        if resp.status_code == 200:
                            logger.info(f"DISPATCHER | Auto-discovered sitemap at common path: {sitemap_url}")
                            return sitemap_url
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"DISPATCHER | Error during sitemap auto-discovery for {url}: {e}")

        return None

    async def ingest(self, source: str, **kwargs) -> AdapterResult:
        """
        Routes the source to the right adapter based on prefix or extension.
        """
        source_lower = source.lower()

        result = None
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            if source_lower.endswith(".xml") or "sitemap" in source_lower:
                logger.info(f"DISPATCHER | Routing to SitemapAdapter | source={source!r}")
                result = await self.sitemap_adapter.ingest(source, **kwargs)
                if not result or not result.documents:
                    logger.warning(f"DISPATCHER | SitemapAdapter returned 0 docs for {source!r}. Falling back to WebAdapter.")
                    result = await self.web_adapter.ingest(source, **kwargs)
            else:
                content_type = await self._detect_http_content_type(source)
                logger.info(f"DISPATCHER | URL detected content_type={content_type!r} source={source!r}")
                if content_type in ("pdf", "txt", "md"):
                    result = await self.universal_adapter.ingest(source, **kwargs)
                else:
                    # Attempt sitemap auto-discovery before falling back to single-page web adapter
                    discovered_sitemap = await self._discover_sitemap(source)
                    if discovered_sitemap:
                        # Auto-extract implicit filter from original URL if user didn't provide one
                        from urllib.parse import urlparse, parse_qs
                        parsed_source = urlparse(source)
                        qs = parse_qs(parsed_source.query)
                        
                        if "filter" not in qs and "filter_prefix" not in kwargs:
                            if parsed_source.path and len(parsed_source.path) > 1: # Ignore "/"
                                kwargs["filter_prefix"] = parsed_source.path
                                logger.info(f"DISPATCHER | Auto-extracted implicit filter prefix '{parsed_source.path}' from original URL")

                        logger.info(f"DISPATCHER | Routing discovered sitemap to SitemapAdapter | source={discovered_sitemap!r}")
                        result = await self.sitemap_adapter.ingest(discovered_sitemap, **kwargs)
                        if not result or not result.documents:
                            logger.warning(f"DISPATCHER | SitemapAdapter returned 0 docs for {discovered_sitemap!r}. Falling back to WebAdapter.")
                            result = await self.web_adapter.ingest(source, **kwargs)
                    else:
                        result = await self.web_adapter.ingest(source, **kwargs)
        elif any(source_lower.endswith(ext) for ext in [".pdf", ".docx", ".md", ".txt"]):
            ext = source_lower.rsplit(".", 1)[-1]
            logger.info(f"DISPATCHER | File upload routed to UniversalAdapter | ext={ext!r} source={source!r}")
            result = await self.universal_adapter.ingest(source, **kwargs)
        else:
            logger.error(f"DISPATCHER | Unsupported input format for source: {source!r}")
            return AdapterResult(documents=[], visual_chunks=[])

        if isinstance(result, list):
            return AdapterResult(documents=result, visual_chunks=[])
        return result

