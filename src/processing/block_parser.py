import hashlib
import re
from typing import List
from src.processing.models import Block, BlockMetrics

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
        elif content.startswith("- ") or content.startswith("* ") or re.match(r"^\d+\.\s", content):
            metrics.is_list = True

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
        code_blocks = {}
        tables = {}

        def repl_code(m):
            ph = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks[ph] = m.group(0)
            return ph

        def repl_table(m):
            ph = f"__TABLE_BLOCK_{len(tables)}__"
            tables[ph] = m.group(0)
            return ph

        content_safe = re.sub(r"```.*?```", repl_code, markdown, flags=re.DOTALL)
        table_pattern = r"(?:(?:^|\n)\|[^\n]+\|)+\n?"
        content_safe = re.sub(table_pattern, repl_table, content_safe)

        blocks = []
        raw_lines = content_safe.split("\n")
        current_block_lines: List[str] = []

        def flush_current():
            content = "\n".join(current_block_lines).strip()
            if content:
                if content in code_blocks:
                    blocks.append(BlockParser.create_block(code_blocks[content]))
                elif content in tables:
                    blocks.append(BlockParser.create_block(tables[content]))
                else:
                    for ph, code in code_blocks.items():
                        content = content.replace(ph, code)
                    for ph, table in tables.items():
                        content = content.replace(ph, table)
                    blocks.append(BlockParser.create_block(content))
            current_block_lines.clear()

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                flush_current()
                continue

            if stripped.startswith("#"):
                flush_current()
                for ph, code in code_blocks.items():
                    stripped = stripped.replace(ph, code)
                blocks.append(BlockParser.create_block(stripped))
                continue

            if stripped in code_blocks or stripped in tables:
                flush_current()
                if stripped in code_blocks:
                    blocks.append(BlockParser.create_block(code_blocks[stripped]))
                else:
                    blocks.append(BlockParser.create_block(tables[stripped]))
                continue

            current_block_lines.append(line)

        flush_current()
        total_blocks = len(blocks)
        for i, block in enumerate(blocks):
            block.metrics.position_ratio = i / total_blocks if total_blocks > 0 else 0.0

        return blocks
