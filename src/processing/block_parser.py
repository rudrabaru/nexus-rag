import hashlib
import re
from typing import List
from src.processing.models import Block, BlockMetrics
from src.protected_markdown import protect

class BlockParser:
    @staticmethod
    def hash_content(content: str) -> str:
        normalized = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"[\1]", content)
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def create_block(content: str) -> Block:
        metrics = BlockMetrics()
        if content.startswith("```") and content.endswith("```"):
            metrics.is_code = True
        elif content.startswith("#"):
            metrics.is_heading = True
        elif "|---|" in content or "|---" in content or "---|" in content:
            metrics.is_table = True

        words = re.findall(r"\b\w+\b", content.lower())
        metrics.word_count = len(words)
        metrics.unique_word_ratio = len(set(words)) / len(words) if words else 0.0

        links = re.findall(r"\[([^\]]+)\]\([^\)]+\)", content)
        metrics.link_count = len(links)
        link_text_length = sum(len(text) for text in links)
        content_length = len(content)
        if content_length > 0:
            metrics.link_density = link_text_length / content_length

        return Block(
            content=content, content_hash=BlockParser.hash_content(content), metrics=metrics
        )

    @staticmethod
    def parse_blocks(markdown: str) -> List[Block]:
        """Blocks in reading order: headings, code, tables and blank-line separated paragraphs."""
        safe, restore = protect(markdown)
        blocks: List[Block] = []
        paragraph: List[str] = []

        def add(content: str) -> None:
            content = restore(content).strip()
            if content:
                blocks.append(BlockParser.create_block(content))

        def flush() -> None:
            add("\n".join(paragraph))
            paragraph.clear()

        for line in safe.split("\n"):
            stripped = line.strip()
            if not stripped:
                flush()
            elif stripped.startswith("#") or stripped.startswith("\x00"):  # a heading, or a whole code block / table
                flush()
                add(stripped)
            else:
                paragraph.append(line)

        flush()
        for position, block in enumerate(blocks):
            block.metrics.position_ratio = position / len(blocks)
        return blocks
