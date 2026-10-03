"""
Document chunking with semantic boundary preservation and overlap.

This module implements the core chunking algorithm:
1. Parse document into sections based on heading hierarchy.
2. Inside sections, split into atomic blocks (Code, Tables, Paragraphs).
3. Group blocks into chunks respecting token budget (soft limit 600, max 800); a block over the
   embedding limit is split, never truncated.
4. Create overlaps between consecutive chunks.
5. Generate chunk metadata.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List

from .metadata import ChunkMetadata, ChunkingConfig
from .tokenizer import TokenCounter
from .blocks import extract_blocks
from .parsers import parse_sections
from .merger import merge_tiny_chunks
from .heuristics import get_overlap_blocks, build_chunk_metadata

logger = logging.getLogger(__name__)


@dataclass
class ChunkFailure:
    url: str
    reason: str


class DocumentChunker:
    """
    Chunks documents into semantically meaningful pieces with token-based sizing.
    """

    def __init__(
        self, config: ChunkingConfig = None, token_counter: TokenCounter = None
    ):
        self.config = config or ChunkingConfig()
        self.token_counter = token_counter or TokenCounter()

        # A block is split to leave room for the overlap that is prepended to the chunk after it, so
        # no chunk can exceed the embedding limit.
        self.max_block_tokens = self.config.embedding_hard_limit - self.config.overlap
        self.failures: List[ChunkFailure] = []

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

        chunks = self._split_content(content, url, title, doc_name)
        logger.debug(f"Created {len(chunks)} chunks from {url}")
        return chunks

    def chunk_batch(self, docs: List[dict]) -> List[ChunkMetadata]:
        """
        One unchunkable page must not sink a whole sitemap job, so a failing document is skipped and
        recorded in `failures`; the caller reports it on the job and fails the job only when nothing
        was produced.
        """
        self.failures = []
        all_chunks = []
        for i, doc in enumerate(docs, 1):
            try:
                all_chunks.extend(self.chunk_document(doc))
            except Exception as e:
                logger.exception(f"Error chunking {doc.get('url')}; the document is skipped")
                self.failures.append(ChunkFailure(doc.get("url", "unknown"), f"{type(e).__name__}: {e}"))
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(docs)} documents, {len(all_chunks)} chunks so far")
        logger.info(
            f"CHUNK | {len(docs)} documents -> {len(all_chunks)} chunks | "
            f"tokens {sum(c.token_count for c in all_chunks)} | "
            f"oversized {sum(c.oversized_chunk for c in all_chunks)} | "
            f"merged from tiny {sum(c.tiny_chunk_merged for c in all_chunks)} | failed {len(self.failures)}"
        )
        return all_chunks

    def _extract_doc_name(self, url: str) -> str:
        # Use an MD5 hash of the full URL to guarantee global uniqueness and prevent collisions across generic documentation sites.
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    def _split_content(
        self, content: str, url: str, title: str, doc_name: str
    ) -> List[ChunkMetadata]:

        sections = parse_sections(content)
        chunks = []
        chunk_index = 0

        for section in sections:
            blocks = extract_blocks(section.text, self.token_counter, self.max_block_tokens)

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
                    chunk = build_chunk_metadata(
                        current_chunk_blocks, chunk_index, url, title, doc_name, section, self.config
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
                        block.block_type in ["code", "table"]
                        and current_tokens + block.token_count
                        <= self.config.max_chunk_tokens
                    ):
                        pass
                    else:
                        chunk = build_chunk_metadata(
                            current_chunk_blocks,
                            chunk_index,
                            url,
                            title,
                            doc_name,
                            section,
                            self.config,
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1

                        overlap_blocks = get_overlap_blocks(current_chunk_blocks, self.config.overlap)
                        current_chunk_blocks = overlap_blocks
                        current_tokens = sum(b.token_count for b in overlap_blocks)

                current_chunk_blocks.append(block)
                current_tokens += block.token_count

            if current_chunk_blocks:
                chunk = build_chunk_metadata(
                    current_chunk_blocks, chunk_index, url, title, doc_name, section, self.config
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1


        return merge_tiny_chunks(chunks, self.config)
