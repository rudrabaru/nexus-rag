"""
Document chunking with semantic boundary preservation and overlap.

This module implements the core chunking algorithm:
1. Parse document into sections based on heading hierarchy.
2. Inside sections, split into atomic blocks (Code, Tables, Paragraphs).
3. Group blocks into chunks respecting token budget (soft limit 600, max 800).
4. Create overlaps between consecutive chunks.
5. Generate chunk metadata.
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .metadata import ChunkMetadata, ChunkingConfig
from .tokenizer import TokenCounter, TokenBudget
from .models import Section, Block
from .parsers import parse_sections, extract_blocks
from .merger import merge_tiny_chunks

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks documents into semantically meaningful pieces with token-based sizing.
    """

    def __init__(
        self, config: ChunkingConfig = None, token_counter: TokenCounter = None
    ):
        self.config = config or ChunkingConfig()
        self.token_counter = token_counter or TokenCounter()
        self.token_budget = TokenBudget(
            self.config.chunk_size, self.config.overlap, self.token_counter
        )

        self.stats = {
            "total_documents_processed": 0,
            "total_chunks_generated": 0,
            "oversized_chunks": 0,
            "tiny_chunks_merged": 0,
            "content_types": {"text": 0, "code": 0, "table": 0, "mixed": 0},
        }

        logger.info(
            f"DocumentChunker initialized: {self.config.chunk_size} tokens, "
            f"{self.config.overlap} overlap"
        )

    def chunk_document(self, doc_dict: dict) -> List[ChunkMetadata]:
        url = doc_dict.get("url", "unknown")
        title = doc_dict.get("title", "Untitled")
        content = doc_dict.get("markdown_content", "")

        if not content.strip():
            logger.warning(f"Empty content for {url}")
            return []

        doc_name = self._extract_doc_name(url)

        try:
            chunks = self._split_content(content, url, title, doc_name)
            # Second pass: set total_chunks and update stats
            total = len(chunks)
            for c in chunks:
                c.total_chunks = total
                self.stats["total_chunks_generated"] += 1
                if c.oversized_chunk:
                    self.stats["oversized_chunks"] += 1
                if c.tiny_chunk_merged:
                    self.stats["tiny_chunks_merged"] += 1
                if c.content_type in self.stats["content_types"]:
                    self.stats["content_types"][c.content_type] += 1
                else:
                    self.stats["content_types"][c.content_type] = 1

            self.stats["total_documents_processed"] += 1
            logger.debug(f"Created {len(chunks)} chunks from {url}")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking {url}: {e}")
            import traceback

            traceback.print_exc()
            return []

    def chunk_batch(self, docs: List[dict]) -> List[ChunkMetadata]:
        all_chunks = []
        for i, doc in enumerate(docs, 1):
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            if i % 10 == 0:
                logger.info(
                    f"Processed {i}/{len(docs)} documents, "
                    f"{len(all_chunks)} chunks so far"
                )
        logger.info(
            f"Completed chunking {len(docs)} documents, total chunks: {len(all_chunks)}"
        )
        return all_chunks

    def _extract_doc_name(self, url: str) -> str:
        # Use an MD5 hash of the full URL to guarantee global uniqueness
        # and prevent collisions across generic documentation sites.
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    def _split_content(
        self, content: str, url: str, title: str, doc_name: str
    ) -> List[ChunkMetadata]:

        sections = parse_sections(content)
        chunks = []
        chunk_index = 0
        char_offset = 0

        for section in sections:
            blocks = extract_blocks(section.text, char_offset, self.token_counter)

            current_chunk_blocks = []
            current_tokens = 0

            for i, block in enumerate(blocks):
                if (
                    (block.block_type in ["code", "table"])
                    and current_chunk_blocks
                    and (
                        current_tokens + block.token_count
                        > self.config.max_chunk_tokens
                    )
                ):
                    chunk = self._create_chunk_from_blocks(
                        current_chunk_blocks, chunk_index, url, title, doc_name, section
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk_blocks = []
                    current_tokens = 0

                if (
                    current_tokens + block.token_count > self.config.chunk_size
                    and current_chunk_blocks
                ):
                    if (
                        current_tokens + block.token_count
                        <= self.config.max_chunk_tokens
                    ):
                        pass  # keep code with explanation
                    else:
                        chunk = self._create_chunk_from_blocks(
                            current_chunk_blocks,
                            chunk_index,
                            url,
                            title,
                            doc_name,
                            section,
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1

                        overlap_blocks = self._get_overlap_blocks(current_chunk_blocks)
                        current_chunk_blocks = overlap_blocks
                        current_tokens = sum(b.token_count for b in overlap_blocks)

                current_chunk_blocks.append(block)
                current_tokens += block.token_count

            if current_chunk_blocks:
                chunk = self._create_chunk_from_blocks(
                    current_chunk_blocks, chunk_index, url, title, doc_name, section
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

            char_offset += len(section.text) + 1

        return merge_tiny_chunks(chunks, self.config)

    def _get_overlap_blocks(self, blocks: List[Block]) -> List[Block]:
        overlap_tokens = 0
        overlap_blocks = []
        for block in reversed(blocks):
            if block.block_type in ["code", "table"]:
                break
            if overlap_tokens + block.token_count > self.config.overlap:
                break
            overlap_blocks.insert(0, block)
            overlap_tokens += block.token_count
        return overlap_blocks

    def _create_chunk_from_blocks(
        self,
        blocks: List[Block],
        chunk_index: int,
        url: str,
        title: str,
        doc_name: str,
        section: Section,
    ) -> Optional[ChunkMetadata]:

        if not blocks:
            return None

        text = "\n\n".join(b.text for b in blocks).strip()
        if not text:
            return None

        token_count = sum(b.token_count for b in blocks)

        heading_match = re.match(r"^#+\s+(.+?)(?:\n|$)", text)
        starts_with_heading = heading_match is not None
        heading = heading_match.group(1) if heading_match else None

        contains_code = any(b.block_type == "code" for b in blocks)
        contains_table = any(b.block_type == "table" for b in blocks)

        content_type = "mixed"
        if all(b.block_type == "text" for b in blocks):
            content_type = "text"
        elif all(b.block_type == "code" for b in blocks):
            content_type = "code"
        elif all(b.block_type == "table" for b in blocks):
            content_type = "table"

        code_languages = []
        if contains_code:
            for b in blocks:
                if b.block_type == "code":
                    lang_match = re.search(r"```(\w+)", b.text)
                    if lang_match:
                        lang = lang_match.group(1)
                        if lang not in code_languages:
                            code_languages.append(lang)

        chunk_id = f"{doc_name}_chunk_{chunk_index:03d}"

        return ChunkMetadata(
            chunk_id=chunk_id,
            source_url=url,
            source_document=doc_name,
            title=title,
            heading_path=section.heading_path,
            section_title=section.title,
            chunk_index=chunk_index,
            total_chunks=0,
            chunk_text=text,
            token_count=token_count,
            char_start=blocks[0].char_start,
            char_end=blocks[-1].char_start + len(blocks[-1].text),
            starts_with_heading=starts_with_heading,
            heading=heading,
            contains_code=contains_code,
            code_languages=code_languages if code_languages else None,
            contains_table=contains_table,
            content_type=content_type,
            document_version=self.config.source_version,
            chunk_version=self.config.output_version,
            table_chunk=(content_type == "table"),
            oversized_chunk=(token_count > self.config.max_chunk_tokens),
            tiny_chunk_merged=False,
            created_at=datetime.utcnow(),
        )
