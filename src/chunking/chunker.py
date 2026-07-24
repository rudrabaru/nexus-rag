"""
Document chunking with semantic boundary preservation and overlap.

This module implements the core chunking algorithm:
1. Parse document into sections based on heading hierarchy.
2. Inside sections, split into atomic blocks (Code, Tables, Paragraphs).
3. Group blocks into chunks respecting token budget (soft limit 600, max 800).
4. Create overlaps between consecutive chunks.
5. Generate chunk metadata.
"""

import logging
import hashlib
from typing import List

from .metadata import ChunkMetadata, ChunkingConfig
from .tokenizer import TokenCounter, TokenBudget
from .parsers import parse_sections, extract_blocks
from .merger import merge_tiny_chunks
from .heuristics import get_overlap_blocks, build_chunk_metadata

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
            "total_tokens_generated": 0,
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
            EMBEDDING_HARD_LIMIT = 2000  # text-embedding-004 token limit (tokens ≈ words * 1.3)
            for c in chunks:
                if c.token_count > EMBEDDING_HARD_LIMIT:
                    logger.warning(
                        f"Chunk {c.chunk_id} has {c.token_count} tokens, "
                        f"which exceeds the embedding model limit of {EMBEDDING_HARD_LIMIT}. "
                        f"Truncating to prevent embedding API failure."
                    )
                    # Truncate text to fit, preserving the heading context
                    words = c.chunk_text.split()
                    max_words = int(EMBEDDING_HARD_LIMIT / 1.3)
                    c.chunk_text = " ".join(words[:max_words]) + " [TRUNCATED]"
                    c.token_count = EMBEDDING_HARD_LIMIT

                c.total_chunks = total
                self.stats["total_chunks_generated"] += 1
                self.stats["total_tokens_generated"] += c.token_count
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
                        pass  # keep code with explanation
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

            char_offset += len(section.text) + 1

        return merge_tiny_chunks(chunks, self.config)
