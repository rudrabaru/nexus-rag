"""Token counting with tiktoken (cl100k_base), shared by every chunking step."""
import logging

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0
