import logging
from src.crawling.metadata import AdapterResult
from src.ingestion.adapters.web import WebAdapter
from src.ingestion.adapters.pdf import PDFAdapter
from src.ingestion.adapters.docx import DocxAdapter
from src.ingestion.adapters.markdown import MarkdownAdapter
from src.ingestion.adapters.rst import RSTAdapter

logger = logging.getLogger(__name__)


class IngestionDispatcher:
    """
    Detects the input format and routes it to the appropriate adapter.
    """

    def __init__(self):
        self.web_adapter = WebAdapter()
        self.pdf_adapter = PDFAdapter()
        self.docx_adapter = DocxAdapter()
        self.md_adapter = MarkdownAdapter()
        self.rst_adapter = RSTAdapter()

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
            content_type = await self._detect_http_content_type(source)
            if content_type == "pdf":
                result = await self.pdf_adapter.ingest(source, **kwargs)
            elif content_type in ("txt", "md"):
                result = await self.md_adapter.ingest(source, **kwargs)
            else:
                result = await self.web_adapter.ingest(source, **kwargs)
        elif source_lower.endswith(".pdf"):
            result = await self.pdf_adapter.ingest(source, **kwargs)
        elif source_lower.endswith(".docx"):
            result = await self.docx_adapter.ingest(source, **kwargs)
        elif source_lower.endswith(".rst"):
            result = await self.rst_adapter.ingest(source, **kwargs)
        elif source_lower.endswith(".md") or source_lower.endswith(".txt"):
            result = await self.md_adapter.ingest(source, **kwargs)
        else:
            logger.error(f"Unsupported input format for source: {source}")
            return AdapterResult(documents=[], visual_chunks=[])

        if isinstance(result, list):
            return AdapterResult(documents=result, visual_chunks=[])
        return result
