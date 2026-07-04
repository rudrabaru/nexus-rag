import os
from typing import List
from src.crawling.metadata import CrawledDocument
from src.crawling.crawler import run_crawler
from src.ingestion.base import IngestionAdapter
import logging

logger = logging.getLogger(__name__)


class WebAdapter(IngestionAdapter):
    """
    Adapter for crawling web URLs (both single seed URLs and sitemaps).
    Wraps the existing run_crawler functionality.
    """

    async def ingest(
        self, source: str, extract_visuals: bool = False, **kwargs
    ) -> List[CrawledDocument]:
        """
        source: URL starting with http:// or https://
        kwargs: max_depth, max_pages, etc.
        """
        logger.info(f"WebAdapter crawling: {source}")

        from urllib.parse import urlparse

        parsed = urlparse(source)
        path_lower = parsed.path.lower()
        is_sitemap = "sitemap" in path_lower or path_lower.endswith(".xml")

        max_depth = kwargs.get("max_depth", 1)
        max_pages = kwargs.get("max_pages", 5)

        import sys
        import asyncio

        def run_in_new_loop():
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    run_crawler(
                        start_url=source if not is_sitemap else None,
                        sitemap_url=source if is_sitemap else None,
                        max_depth=max_depth,
                        max_pages=max_pages,
                    )
                )
            finally:
                loop.close()

        docs = await asyncio.to_thread(run_in_new_loop)

        if extract_visuals:
            from bs4 import BeautifulSoup
            import httpx
            from urllib.parse import urljoin
            from src.ingestion.visual_processor import VisualProcessor
            from PIL import Image
            import io

            processor = VisualProcessor()

            async with httpx.AsyncClient() as client:
                for doc in docs:
                    if not doc.raw_html:
                        continue

                    soup = BeautifulSoup(doc.raw_html, "html.parser")
                    images = soup.find_all("img")

                    visual_additions = []
                    for img in images:
                        src = img.get("src")
                        if not src:
                            continue

                        # Handle data URIs or relative links
                        if src.startswith("data:image"):
                            continue

                        img_url = urljoin(doc.url, src)

                        try:
                            resp = await client.get(img_url, timeout=5)
                            if resp.status_code != 200:
                                continue

                            # Stricter filter: > 5KB
                            if len(resp.content) < 5120:
                                continue

                            # Stricter filter: > 100x100 pixels
                            img_pil = Image.open(io.BytesIO(resp.content))
                            width, height = img_pil.size
                            if width < 100 or height < 100:
                                continue

                            desc = await asyncio.to_thread(
                                processor.describe_image, resp.content
                            )
                            if desc:
                                visual_additions.append(
                                    f"> **Visual Element ({img_url}):**\n> {desc}\n"
                                )

                        except Exception as e:
                            logger.warning(
                                f"Failed to process web image {img_url}: {e}"
                            )

                    if visual_additions:
                        doc.markdown_content += (
                            "\n\n## Extracted Visuals\n\n" + "\n".join(visual_additions)
                        )

        return docs
