import os
import httpx
from src.crawling.metadata import CrawledDocument, AdapterResult
from src.ingestion.base import IngestionAdapter
import logging

logger = logging.getLogger(__name__)

class WebAdapter(IngestionAdapter):
    """
    Adapter for crawling web URLs via Jina Reader.
    """

    async def ingest(
        self, source: str, extract_visuals: bool = False, **kwargs
    ) -> AdapterResult:
        """
        source: URL starting with http:// or https://
        kwargs: max_depth, max_pages, etc. (ignored for now via Jina Reader)
        """
        logger.info(f"WebAdapter fetching: {source} via Jina Reader")

        jina_api_key = os.environ.get("JINA_API_KEY")
        if not jina_api_key:
            raise ValueError("JINA_API_KEY environment variable is missing.")

        headers = {
            "Authorization": f"Bearer {jina_api_key}",
            "Accept": "application/json",
            "X-Return-Format": "markdown",
            "X-Timeout": "20",
        }
        if not extract_visuals:
            headers["X-Retain-Images"] = "none"

        import asyncio
        async with httpx.AsyncClient(timeout=30.0) as client:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await client.get(
                        f"https://r.jina.ai/{source}",
                        headers=headers
                    )
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)
            content = response.json()["data"]["content"]
            title = response.json()["data"].get("title", source)

        word_count = len(content.split())
        if word_count < 30:
            logger.warning(
                f"URL {source} returned {word_count} words. "
                f"Possible bot-block, redirect, or empty page."
            )
            raise ValueError(
                "No readable content was extracted from the URL. "
                "The site may require login or block crawlers."
            )

        doc = CrawledDocument(
            url=source,
            title=title,
            markdown_content=content
        )

        return AdapterResult(documents=[doc], visual_chunks=[])
