import json
import time
import logging
import hashlib
from typing import List

from src.retrieving.retriever import RetrievedChunk
from .models import ContextChunk, ContextWindow, GenerationConfig

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from word count.

    Heuristic: tokens ≈ words / 0.75 (English prose averages ~0.75 words/token).
    This is accurate enough for context budget management without requiring a tokenizer.
    """
    word_count = len(text.split())
    return max(1, int(word_count / 0.75))


def _format_heading_path(heading_path: List[str]) -> str:
    """Formats a heading path list into a readable breadcrumb string."""
    if not heading_path:
        return ""
    return " > ".join(heading_path)


class ContextBuilder:
    """
    Assembles a ContextWindow from a list of RetrievedChunks.

    The context window respects the token budget defined in GenerationConfig
    and adds source citation headers to each chunk for LLM grounding.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def build(self, retrieved_chunks: List[RetrievedChunk]) -> ContextWindow:
        """
        Build a token-budgeted context window from retrieved chunks.

        Args:
            retrieved_chunks: Ordered list of chunks from DenseRetriever (highest score first).

        Returns:
            ContextWindow with included/excluded chunk lists and assembled context_text.
        """
        start = time.time()

        # Convert to ContextChunk for cleaner downstream handling
        context_chunks = [self._to_context_chunk(c) for c in retrieved_chunks]

        included: List[ContextChunk] = []
        excluded: List[ContextChunk] = []
        total_tokens = 0
        context_parts: List[str] = []

        seen_fingerprints = set()

        for chunk in context_chunks:
            if chunk.similarity_score < self.config.min_similarity_score:
                excluded.append(chunk)
                logger.debug(
                    f"Excluded chunk {chunk.chunk_id} (score {chunk.similarity_score:.3f} < {self.config.min_similarity_score})"
                )
                continue

            fingerprint = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
            if fingerprint in seen_fingerprints:
                logger.debug(f"Excluded chunk {chunk.chunk_id} (duplicate content)")
                continue

            if total_tokens + chunk.token_estimate > self.config.max_context_tokens:
                excluded.append(chunk)
                logger.debug(
                    f"Excluded chunk {chunk.chunk_id} (budget exceeded: "
                    f"{total_tokens + chunk.token_estimate} > {self.config.max_context_tokens})"
                )
                continue

            seen_fingerprints.add(fingerprint)

            included.append(chunk)
            total_tokens += chunk.token_estimate
            context_parts.append(self._format_chunk(chunk))

        context_text = "\n\n---\n\n".join(context_parts)

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"Context built: {len(included)} chunks included, {len(excluded)} excluded, "
            f"~{total_tokens} tokens. ({elapsed:.1f}ms)"
        )

        return ContextWindow(
            included_chunks=included,
            excluded_chunks=excluded,
            total_context_tokens=total_tokens,
            context_text=context_text,
        )

    def _to_context_chunk(self, chunk: RetrievedChunk) -> ContextChunk:
        """Convert a RetrievedChunk to a ContextChunk with token estimate."""
        # Parse heading_path from metadata (stored as JSON array string)
        raw_path = chunk.metadata.get("heading_path", "")
        try:
            heading_path = json.loads(raw_path) if raw_path else []
            if not isinstance(heading_path, list):
                heading_path = [str(heading_path)]
        except (json.JSONDecodeError, TypeError):
            heading_path = [s.strip() for s in str(raw_path).split(" > ") if s.strip()]

        source_url = chunk.metadata.get("source_url", chunk.source_document)

        return ContextChunk(
            chunk_id=chunk.chunk_id,
            source_url=source_url,
            heading_path=heading_path,
            text=chunk.text,
            similarity_score=chunk.similarity_score,
            token_estimate=_estimate_tokens(chunk.text),
        )

    def _format_chunk(self, chunk: ContextChunk) -> str:
        """
        Format a single chunk for inclusion in the context string.

        Adds a citation header above each chunk so the LLM knows where the
        information originated. This is the key mechanism for grounded answers.
        """
        heading_str = _format_heading_path(chunk.heading_path)
        header = f"[Source: {chunk.source_url}"
        if heading_str:
            header += f" | Section: {heading_str}"
        header += "]"

        return f"{header}\n{chunk.text}"
