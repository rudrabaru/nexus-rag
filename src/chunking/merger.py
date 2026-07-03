from typing import List
from .metadata import ChunkMetadata
from .metadata import ChunkingConfig


def merge_tiny_chunks(
    chunks: List[ChunkMetadata], config: ChunkingConfig
) -> List[ChunkMetadata]:
    """Merges tiny or incomplete chunks into neighboring related chunks."""
    if not chunks:
        return []

    merged = []
    current = chunks[0]

    for next_chunk in chunks[1:]:
        # 1. Semantic Completeness Checks
        lines_current = [
            line for line in current.chunk_text.strip().split("\n") if line.strip()
        ]
        current_is_heading_only = (
            current.starts_with_heading
            and len(lines_current) <= 2
            and current.token_count < 40
        )
        current_is_tiny = current.token_count < 40

        # Note: We don't check next_chunk's incompleteness to force a merge into current.
        # If next_chunk is incomplete (e.g. a heading), it will become `current` on the next iteration,
        # and the text *after* it will be merged into it. Merging a heading into the end of an existing
        # complete chunk is semantically wrong.

        # 2. Content Type Checks
        is_content_clash = (
            current.content_type == "table" and next_chunk.content_type == "table"
        ) or (current.content_type == "code" and next_chunk.content_type == "code")

        # 3. Hierarchy Checks
        # Sibling: same parent path
        is_sibling = False
        if (
            len(current.heading_path) == len(next_chunk.heading_path)
            and len(current.heading_path) > 0
        ):
            if current.heading_path[:-1] == next_chunk.heading_path[:-1]:
                is_sibling = True

        # Parent-Child: next is child of current
        is_parent_child = False
        if len(current.heading_path) < len(next_chunk.heading_path):
            if (
                next_chunk.heading_path[: len(current.heading_path)]
                == current.heading_path
            ):
                is_parent_child = True

        # Same Exact Path
        is_same_path = current.heading_path == next_chunk.heading_path

        # Are they related enough to consider merging?
        is_related = is_same_path or is_parent_child or is_sibling
        if not current.heading_path or not next_chunk.heading_path:
            is_related = True  # Top level elements

        should_merge = False

        if not is_content_clash and is_related:
            # Always merge if current is just a heading, to give it body text
            # We enforce max_chunk_tokens unless current is JUST a tiny heading that MUST be merged.
            if current_is_heading_only:
                should_merge = True
            # Otherwise, merge if current is tiny, or if both are under min threshold, AS LONG AS it fits in max
            elif current_is_tiny or (
                current.token_count < config.min_chunk_tokens
                or next_chunk.token_count < config.min_chunk_tokens
            ):
                if (
                    current.token_count + next_chunk.token_count
                    <= config.max_chunk_tokens
                ):
                    should_merge = True

        if should_merge:
            current.chunk_text += "\n\n" + next_chunk.chunk_text
            current.token_count += next_chunk.token_count
            current.char_end = next_chunk.char_end
            current.contains_code = current.contains_code or next_chunk.contains_code
            current.contains_table = current.contains_table or next_chunk.contains_table
            current.tiny_chunk_merged = True

            if current.content_type != next_chunk.content_type:
                current.content_type = "mixed"

            if current_is_heading_only and len(next_chunk.heading_path) > len(
                current.heading_path
            ):
                current.heading_path = next_chunk.heading_path
                current.section_title = next_chunk.section_title
        else:
            merged.append(current)
            current = next_chunk

    merged.append(current)

    for i, c in enumerate(merged):
        c.chunk_index = i
        c.total_chunks = len(merged)

    return merged
