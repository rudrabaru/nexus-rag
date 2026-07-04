from typing import List
from datetime import datetime
from src.crawling.metadata import CrawledDocument
from src.ingestion.base import IngestionAdapter
import logging

logger = logging.getLogger(__name__)


class MarkdownAdapter(IngestionAdapter):
    """
    Adapter for reading local Markdown files.
    """

    async def ingest(
        self, source: str, extract_visuals: bool = False, **kwargs
    ) -> List[CrawledDocument]:
        logger.info(
            f"MarkdownAdapter reading: {source}, extract_visuals={extract_visuals}"
        )

        try:
            if source.startswith("http://") or source.startswith("https://"):
                import httpx

                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=30
                ) as client:
                    resp = await client.get(source)
                    resp.raise_for_status()
                    markdown_content = resp.text
                    title = source.split("/")[-1] or source
            else:
                with open(source, "r", encoding="utf-8", errors="replace") as f:
                    markdown_content = f.read()
                from pathlib import Path

                title = Path(source).name

            # Normalize line endings
            markdown_content = markdown_content.replace("\r\n", "\n").replace(
                "\r", "\n"
            )

            # Strip YAML front matter if present
            import re

            front_matter_match = re.match(
                r"^\s*---\n(.*?)\n---\n", markdown_content, flags=re.DOTALL
            )
            if front_matter_match:
                import yaml

                try:
                    front_matter = yaml.safe_load(front_matter_match.group(1))
                    if isinstance(front_matter, dict) and "title" in front_matter:
                        title = str(front_matter["title"])
                except Exception as e:
                    logger.warning(f"Failed to parse YAML front matter: {e}")
                markdown_content = markdown_content[front_matter_match.end() :].strip()

            if extract_visuals:
                import re
                import os
                from src.ingestion.visual_processor import VisualProcessor

                processor = VisualProcessor()

                # Regex for markdown images: ![alt](url)
                img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

                def replace_img(match):
                    alt_text = match.group(1)
                    img_path = match.group(2)

                    # Only process local files (skip http)
                    if not img_path.startswith("http"):
                        base_dir = os.path.dirname(os.path.abspath(source))
                        full_img_path = os.path.join(base_dir, img_path)

                        if os.path.exists(full_img_path):
                            try:
                                with open(full_img_path, "rb") as img_file:
                                    img_data = img_file.read()

                                desc = processor.describe_image(img_data)
                                if desc:
                                    return f"\n> **Visual Element ({alt_text or img_path}):**\n> {desc}\n"
                            except Exception as e:
                                logger.warning(
                                    f"Failed to process local image {full_img_path}: {e}"
                                )

                    # If failed, remote, or not found, return original tag
                    return match.group(0)

                markdown_content = img_pattern.sub(replace_img, markdown_content)

            word_count = len(markdown_content.split())

            crawled_doc = CrawledDocument(
                url=source if source.startswith("http") else f"file:///{source}",
                title=title,
                markdown_content=markdown_content,
                crawl_depth=0,
                crawled_at=datetime.utcnow(),
                word_count=word_count,
                status_code=200,
            )

            return [crawled_doc]

        except Exception as e:
            logger.error(f"Failed to read MD {source}: {e}")
            return []
