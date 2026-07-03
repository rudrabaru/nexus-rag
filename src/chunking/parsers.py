import re
from typing import List
from .models import Section, Block
from .tokenizer import TokenCounter


def parse_sections(content: str) -> List[Section]:
    """Parse markdown content into sections based on headings."""
    sections = []
    heading_stack = []  # List of tuples: (level, title)

    lines = content.split("\n")

    current_section = Section(title="Introduction", level=0, heading_path=[])
    current_text = []

    # Regex to capture markdown headings
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    for line in lines:
        match = heading_re.match(line)
        if match:
            # Save current section
            if current_text:
                current_section.text = "\n".join(current_text)
                sections.append(current_section)
                current_text = []
            elif current_section.title == "Introduction" and not sections:
                # Empty introduction, just skip
                pass
            else:
                current_section.text = ""
                sections.append(current_section)

            level = len(match.group(1))
            title = match.group(2).strip()

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
        current_section.text = "\n".join(current_text)
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
                    split_paragraphs.append("\n".join(current_sub))
                    current_sub = [sp]
                    current_len = len(sp)
                else:
                    current_sub.append(sp)
                    current_len += len(sp) + 1
            if current_sub:
                split_paragraphs.append("\n".join(current_sub))
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
            l.strip().startswith("|") and l.strip().endswith("|") for l in lines
        )
        has_separator = any(re.match(r"^\|[-| :]+\|$", l.strip()) for l in lines)
        is_table = has_pipe_row and has_separator

        block_type = "text"
        if is_code:
            block_type = "code"
        elif is_table:
            block_type = "table"

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
