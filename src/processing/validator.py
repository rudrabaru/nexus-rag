import re
from src.processing.models import ProcessedDocument


class ProcessingValidator:

    @staticmethod
    def validate_document(
        raw_content: str, processed_blocks: list, doc: ProcessedDocument
    ):
        """Computes before/after statistics and validates preservation."""
        stats = doc.processing_stats

        # Before metrics
        stats.words_before = len(raw_content.split())
        stats.h1_before = len(re.findall(r"^#\s+.*$", raw_content, re.MULTILINE))
        stats.h2_before = len(re.findall(r"^##\s+.*$", raw_content, re.MULTILINE))
        stats.h3_before = len(re.findall(r"^###\s+.*$", raw_content, re.MULTILINE))
        stats.h4_before = len(re.findall(r"^####\s+.*$", raw_content, re.MULTILINE))
        stats.code_blocks_before = (
            len(re.findall(r"^```", raw_content, re.MULTILINE)) // 2
        )
        stats.tables_before = len(re.findall(r"\|[\s\-:]+\|", raw_content))

        # After metrics
        after_content = "\n\n".join(b.content for b in processed_blocks)
        doc.cleaned_markdown = after_content

        stats.words_after = len(after_content.split())
        stats.h1_after = len(re.findall(r"^#\s+.*$", after_content, re.MULTILINE))
        stats.h2_after = len(re.findall(r"^##\s+.*$", after_content, re.MULTILINE))
        stats.h3_after = len(re.findall(r"^###\s+.*$", after_content, re.MULTILINE))
        stats.h4_after = len(re.findall(r"^####\s+.*$", after_content, re.MULTILINE))
        stats.code_blocks_after = (
            len(re.findall(r"^```", after_content, re.MULTILINE)) // 2
        )
        stats.tables_after = len(re.findall(r"\|[\s\-:]+\|", after_content))

        # Retention
        if stats.words_before > 0:
            stats.retention_percentage = (stats.words_after / stats.words_before) * 100
        else:
            stats.retention_percentage = 0.0

