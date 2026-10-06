from typing import List


class Section:
    """Represents a section under a specific heading."""

    def __init__(self, title: str, level: int, heading_path: List[str]):
        self.title = title
        self.level = level
        self.heading_path = heading_path
        self.text = ""


class Block:
    """Represents an atomic unit of text within a section."""

    def __init__(self, text: str, block_type: str, token_count: int):
        self.text = text
        self.block_type = block_type  # 'text', 'code', 'table'
        self.token_count = token_count
