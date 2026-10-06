"""Token counting with tiktoken (cl100k_base), shared by every chunking step."""
import tiktoken


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """
        Special-token strings such as <|endoftext|> are ordinary text in a document (a page about
        LLMs mentions them), so they are encoded as text. tiktoken refuses them by default, which
        used to make the count 0 and the chunk look empty.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text, disallowed_special=()))
