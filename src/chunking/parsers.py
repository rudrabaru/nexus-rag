import re
from typing import List
from .models import Section, Block
from .tokenizer import TokenCounter

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def _sentence_boundary_split(text: str, max_chars: int = 3000) -> List[str]:
    """
    Split a wall-of-text block on sentence boundaries.
    """
    sentences = _SENTENCE_END.split(text)
    split_chunks = []
    current_chunk = []
    current_len = 0
    
    for s in sentences:
        if current_len + len(s) > max_chars and current_chunk:
            split_chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_len = len(s)
        else:
            current_chunk.append(s)
            current_len += len(s) + 1
            
    if current_chunk:
        split_chunks.append(" ".join(current_chunk))
    return split_chunks

def parse_sections(content: str) -> List[Section]:
    """Parse markdown content into sections based on headings."""
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

    # Pre-extract code blocks (```...```)
    content_safe = re.sub(r"```.*?```", repl_code, content, flags=re.DOTALL)

    # Pre-extract markdown tables (lines with |)
    table_pattern = r"(?:(?:^|\n)\|[^\n]+\|)+\n?"
    content_safe = re.sub(table_pattern, repl_table, content_safe)

    sections = []
    heading_stack = []  # List of tuples: (level, title)

    lines = content_safe.split("\n")

    current_section = Section(title="", level=0, heading_path=[])
    current_text = []

    # Regex to capture markdown headings
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    
    def inject_placeholders(text):
        for ph, code in code_blocks.items():
            text = text.replace(ph, code)
        for ph, table in tables.items():
            text = text.replace(ph, table)
        return text

    for line in lines:
        match = heading_re.match(line)
        if match:
            # Save current section
            if current_text:
                current_section.text = inject_placeholders("\n".join(current_text))
                sections.append(current_section)
                current_text = []
            elif current_section.title == "" and not sections and not current_text:
                # Empty root section, just skip
                pass
            else:
                current_section.text = ""
                sections.append(current_section)

            level = len(match.group(1))
            title = match.group(2).strip()
            
            title = inject_placeholders(title)

            # Update stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            heading_path = [item[1] for item in heading_stack]
            current_section = Section(
                title=title, level=level, heading_path=heading_path
            )
            current_text.append(line)  # Include the heading in the section text
        else:
            current_text.append(line)

    if current_text:
        current_section.text = inject_placeholders("\n".join(current_text))
        sections.append(current_section)

    return [s for s in sections if s.text.strip()]


def extract_blocks(
    text: str, char_offset_base: int, token_counter: TokenCounter
) -> List[Block]:
    """Extract atomic blocks (Code, Table, Paragraph) from a section's text."""
    blocks = []

    raw_paragraphs = re.split(r"\n\n+", text.strip())

    # Fallback: if a paragraph is massive, split it by single newlines or periods
    split_paragraphs = []
    for p in raw_paragraphs:
        if len(p) > 5000 and not p.startswith("```"):  # ~1000 words
            # Try to split by single newline first
            sub_p = p.split("\n")
            current_sub = []
            current_len = 0
            for sp in sub_p:
                if current_len + len(sp) > 3000 and current_sub:
                    chunk_text = "\n".join(current_sub)
                    if len(chunk_text) > 4000:
                        split_paragraphs.extend(_sentence_boundary_split(chunk_text))
                    else:
                        split_paragraphs.append(chunk_text)
                    current_sub = [sp]
                    current_len = len(sp)
                else:
                    current_sub.append(sp)
                    current_len += len(sp) + 1
            if current_sub:
                chunk_text = "\n".join(current_sub)
                if len(chunk_text) > 4000:
                    split_paragraphs.extend(_sentence_boundary_split(chunk_text))
                else:
                    split_paragraphs.append(chunk_text)
        else:
            split_paragraphs.append(p)

    current_char = char_offset_base

    for para in split_paragraphs:
        if not para.strip():
            continue

        # Classify
        is_code = para.strip().startswith("```")

        lines = para.split("\n")
        has_pipe_row = any(
            line.strip().startswith("|") and line.strip().endswith("|") for line in lines
        )
        has_separator = any(re.match(r"^\|[-| :]+\|$", line.strip()) for line in lines)
        is_table = has_pipe_row and has_separator

        block_type = "text"
        if is_code:
            block_type = "code"
        elif is_table:
            block_type = "table"
        # 5E.2 Table Reconstruction
        is_orphaned_table = False
        if not is_code and not is_table:
            pipe_lines = [line for line in lines if line.strip().startswith("|") and line.strip().endswith("|")]
            if len(pipe_lines) >= 2 and len(pipe_lines) >= len([line for line in lines if line.strip()]) * 0.8:
                is_orphaned_table = True

        if is_orphaned_table and blocks and blocks[-1].block_type == "table":
            # Check column counts
            prev_table_lines = blocks[-1].text.strip().split("\n")
            prev_cols = prev_table_lines[0].count("|")
            this_cols = pipe_lines[0].count("|")
            
            if prev_cols == this_cols:
                # Merge it!
                blocks[-1].text += "\n" + para
                blocks[-1].token_count = token_counter.count_tokens(blocks[-1].text)
                current_char += len(para) + 2
                continue
                
        token_count = token_counter.count_tokens(para)
        blocks.append(
            Block(
                text=para,
                block_type=block_type,
                token_count=token_count,
                char_start=current_char,
            )
        )

        # Approximate char advance
        current_char += len(para) + 2

    return blocks
