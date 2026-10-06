"""
Choosing which chunks to write questions from.

- Chunks with identical text form one group and one question source. The question's ground truth
  is every chunk of the group: any of them answers it equally well, and naming only one would
  mark a correct retrieval wrong. Nothing is dropped for being a duplicate.
- Groups are interleaved across documents (round-robin, seeded), so a long document cannot
  dominate the set and the same seed gives the same order on the same index.
- Which chunks hold a question worth asking is the model's call, not a rule of ours (the prompt lets
  it abstain), so no length or keyword filter decides what the corpus is "worth" testing.
"""
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.db.schema import chunks


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    doc_id: str
    source: str  # the URL, or the file name of an upload
    section_title: str
    heading_path: tuple
    text: str
    contains_code: bool
    contains_table: bool

    @property
    def leaf_heading(self) -> str:
        return self.section_title or (self.heading_path[-1] if self.heading_path else "")


@dataclass(frozen=True)
class ChunkGroup:
    """Chunks with the same text: one question source, and all of them are its ground truth."""

    chunks: tuple  # SourceChunk, the first being the representative the question is written from

    @property
    def representative(self) -> SourceChunk:
        return self.chunks[0]

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def documents(self) -> List[str]:
        return sorted({c.source for c in self.chunks})

    @property
    def headings(self) -> List[str]:
        return sorted({c.leaf_heading for c in self.chunks if c.leaf_heading})


def load_chunks(engine: Engine, tenant_id: str, index_id: str) -> List[SourceChunk]:
    """One tenant's chunks of one index (every index holds the same text; one is enough)."""
    stmt = select(
        chunks.c.chunk_id, chunks.c.doc_id, chunks.c.source_document, chunks.c.source_url, chunks.c.section_title,
        chunks.c.heading_path, chunks.c.chunk_text, chunks.c.contains_code, chunks.c.contains_table,
    ).where(chunks.c.tenant_id == tenant_id, chunks.c.index_id == index_id).order_by(chunks.c.doc_id, chunks.c.chunk_id)
    with engine.connect() as conn:
        return [
            SourceChunk(
                chunk_id=r.chunk_id, doc_id=r.doc_id, source=r.source_url or r.source_document,
                section_title=r.section_title or "", heading_path=tuple(r.heading_path or []), text=r.chunk_text,
                contains_code=r.contains_code, contains_table=r.contains_table,
            )
            for r in conn.execute(stmt)
        ]


def group_identical(source_chunks: List[SourceChunk]) -> List[ChunkGroup]:
    by_text: Dict[str, List[SourceChunk]] = defaultdict(list)
    for chunk in source_chunks:
        by_text[hashlib.sha256(" ".join(chunk.text.split()).encode("utf-8")).hexdigest()].append(chunk)
    return [ChunkGroup(tuple(members)) for members in by_text.values()]


def interleave_by_document(groups: List[ChunkGroup], seed: int) -> List[ChunkGroup]:
    """Round-robin over documents, in a seeded random order within and across documents."""
    rng = random.Random(seed)
    per_document: Dict[str, List[ChunkGroup]] = defaultdict(list)
    for group in sorted(groups, key=lambda g: g.chunk_ids[0]):
        per_document[group.representative.doc_id].append(group)
    queues = [per_document[doc_id] for doc_id in sorted(per_document)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    ordered = []
    while any(queues):
        ordered.extend(queue.pop() for queue in queues if queue)
    return ordered
