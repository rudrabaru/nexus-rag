from typing import List
from datetime import datetime
from src.crawling.metadata import CrawledDocument
from src.ingestion.base import IngestionAdapter
import logging
import os

logger = logging.getLogger(__name__)


class RSTAdapter(IngestionAdapter):
    """
    Adapter for reading local reStructuredText (.rst) files and converting them to Markdown.
    Requires pypandoc.
    """

    async def ingest(
        self, source: str, extract_visuals: bool = False, **kwargs
    ) -> List[CrawledDocument]:
        logger.info(f"RSTAdapter reading: {source}")

        try:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                rst_content = f.read()

            try:
                import pypandoc

                markdown_content = pypandoc.convert_text(
                    rst_content, "md", format="rst"
                )
            except ImportError:
                logger.warning("pypandoc not installed, treating RST as raw text.")
                markdown_content = rst_content

            word_count = len(markdown_content.split())
            from pathlib import Path

            title = Path(source).name

            crawled_doc = CrawledDocument(
                url=f"file:///{source}",
                title=title,
                markdown_content=markdown_content,
                crawl_depth=0,
                crawled_at=datetime.utcnow(),
                word_count=word_count,
                status_code=200,
            )

            return [crawled_doc]

        except Exception as e:
            logger.error(f"Failed to read RST {source}: {e}")
            return []
