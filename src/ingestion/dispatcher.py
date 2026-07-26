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

