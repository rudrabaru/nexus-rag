"""
Is a retrieved chunk relevant to a query? Decided from document identity and headings, with the
same rules the previous harness used, so results stay comparable across the two:

- document: an acceptable document is a substring of the chunk's source URL, or its
  alphanumeric tokens appear as a contiguous run in the URL's tokens ("3.13.html" matches
  ".../whatsnew/3.13.html");
- heading: when the query lists acceptable headings, one of them must appear (same token rule)
  in the chunk's section title or heading path.

"exact" = right document and heading; "partial" = right document, heading constraint unmet.
Both count as relevant for rank; "exact" also sets exact_rank.

With relevance="chunk" a chunk is relevant only when its id is one of the query's
source_chunk_ids: a stricter test that can tell two chunks of the right section apart, at the
price of ground truth that is tied to this chunking (see Phase 6). Only "exact" exists there.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from src.evaluation.dataset import EvaluationQuery
from src.retrieving.models import RetrievedChunk

EXACT, PARTIAL, NONE = "exact", "partial", "none"


def tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def contains_run(needle: List[str], haystack: List[str]) -> bool:
    n = len(needle)
    return n == 0 or any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def heading_path(chunk: RetrievedChunk) -> List[str]:
    return list(chunk.heading_path)


def match(chunk: RetrievedChunk, query: EvaluationQuery, relevance: str = "document") -> str:
    if relevance == "chunk":
        return EXACT if chunk.chunk_id in query.source_chunk_ids else NONE
    source = chunk.metadata.get("source_url") or chunk.source_document
    document_ok = any(
        acceptable in source or contains_run(tokens(acceptable), tokens(source))
        for acceptable in query.acceptable_documents
    )
    if not document_ok:
        return NONE
    if not query.acceptable_headings:
        return EXACT
    where = [tokens(chunk.metadata.get("section_title") or ""), tokens(" > ".join(heading_path(chunk)))]
    heading_ok = any(contains_run(tokens(h), w) for h in query.acceptable_headings for w in where)
    return EXACT if heading_ok else PARTIAL


@dataclass
class Judgement:
    matches: List[str]  # one per retrieved chunk, in rank order
    rank: Optional[int]  # 1-based rank of the first relevant chunk, None if none
    exact_rank: Optional[int]


def judge(chunks: List[RetrievedChunk], query: EvaluationQuery, relevance: str = "document") -> Judgement:
    matches = [match(c, query, relevance) for c in chunks]
    rank = next((i + 1 for i, m in enumerate(matches) if m != NONE), None)
    exact_rank = next((i + 1 for i, m in enumerate(matches) if m == EXACT), None)
    return Judgement(matches, rank, exact_rank)
