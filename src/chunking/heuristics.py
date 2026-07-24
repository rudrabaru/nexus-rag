import re
from typing import List, Optional
from datetime import datetime, timezone

from .metadata import ChunkMetadata, ChunkingConfig
from .models import Section, Block


def get_overlap_blocks(blocks: List[Block], overlap_budget: int) -> List[Block]:
    """
    Returns blocks from the end of the list that fit within the overlap token budget.
    Stops if it hits a code or table block to avoid fragmenting structured data.
    """
    overlap_tokens = 0
    overlap_blocks = []
    for block in reversed(blocks):
        if block.block_type in ["code", "table"]:
            break
        if overlap_tokens + block.token_count > overlap_budget:
            break
        overlap_blocks.insert(0, block)
        overlap_tokens += block.token_count
    return overlap_blocks


def build_chunk_metadata(
    blocks: List[Block],
    chunk_index: int,
    url: str,
    title: str,
    doc_name: str,
    section: Section,
    config: ChunkingConfig,
) -> Optional[ChunkMetadata]:
    """
    Constructs a ChunkMetadata object from a list of blocks, applying heuristics to
    determine content type, code languages, and heading presence.
    """
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
        source_document=title if title else url,
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
        document_version=config.source_version,
        chunk_version=config.output_version,
        table_chunk=(content_type == "table"),
        oversized_chunk=(token_count > config.max_chunk_tokens),
        tiny_chunk_merged=False,
        created_at=datetime.now(timezone.utc),
    )
