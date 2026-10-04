import hashlib
import logging
import time
from typing import Dict, List

from src.retrieving.models import RetrievedChunk
from src.tokens import estimate_tokens

from .models import ContextChunk, ContextWindow, GenerationConfig

logger = logging.getLogger(__name__)

BELOW_SCORE, DUPLICATE, OVER_BUDGET = "score", "duplicate", "budget"


def _format_heading_path(heading_path: List[str]) -> str:
    return " > ".join(heading_path)


class ContextBuilder:
    """
    Assembles a ContextWindow from retrieved chunks: highest score first, within the token budget, each
    chunk under a citation header so the model knows where the information came from.

    Every chunk that is not included is recorded in `excluded_chunks` with its reason, so a run can
    always say whether context was lost to a score floor, a repeated chunk or the budget.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def build(self, retrieved_chunks: List[RetrievedChunk]) -> ContextWindow:
        start = time.time()
        included: List[ContextChunk] = []
        excluded: List[ContextChunk] = []
        reasons: Dict[str, str] = {}
        parts: List[str] = []
        total_tokens = 0
        seen = set()

        for retrieved in retrieved_chunks:
            chunk = self._to_context_chunk(retrieved)
            fingerprint = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
            if chunk.similarity_score < self.config.min_similarity_score:
                reason = BELOW_SCORE
            elif fingerprint in seen:
                reason = DUPLICATE
            elif total_tokens + chunk.token_estimate > self.config.max_context_tokens:
                reason = OVER_BUDGET
            else:
                seen.add(fingerprint)
                included.append(chunk)
                total_tokens += chunk.token_estimate
                parts.append(self._format_chunk(chunk))
                continue
            excluded.append(chunk)
            reasons[chunk.chunk_id] = reason
            logger.debug(f"Excluded chunk {chunk.chunk_id} ({reason}, score {chunk.similarity_score:.3f})")

        logger.info(
            f"Context built: {len(included)} chunks included, {len(excluded)} excluded, "
            f"~{total_tokens} tokens. ({(time.time() - start) * 1000:.1f}ms)"
        )
        return ContextWindow(
            included_chunks=included,
            excluded_chunks=excluded,
            exclusion_reasons=reasons,
            total_context_tokens=total_tokens,
            context_text="\n\n---\n\n".join(parts),
        )

    def _to_context_chunk(self, chunk: RetrievedChunk) -> ContextChunk:
        source_url = chunk.source_url or chunk.metadata.get("source_url") or chunk.source_document
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            source_url=source_url,
            heading_path=list(chunk.heading_path),
            text=chunk.text,
            similarity_score=chunk.similarity_score,
            token_estimate=chunk.token_count or estimate_tokens(chunk.text),  # the chunker's real count when the row has one
        )

    def _format_chunk(self, chunk: ContextChunk) -> str:
        """The chunk under a `[Source: url | Section: path]` header; this is the mechanism for grounded, citable answers."""
        header = f"[Source: {chunk.source_url}"
        if chunk.heading_path:
            header += f" | Section: {_format_heading_path(chunk.heading_path)}"
        return f"{header}]\n{chunk.text}"
