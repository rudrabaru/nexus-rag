import hashlib
import re
from typing import List, Dict
from src.processing.models import Block, BlockMetrics


class DocumentCleaner:
    def __init__(self, total_documents: int):
        self.total_documents = total_documents
        self.block_document_counts: Dict[str, int] = {}

    def _hash_content(self, content: str) -> str:
        # Strip URLs from markdown links [text](url) -> [text]
        normalized = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"[\1]", content)
        # Strip dates like YYYY-MM-DD
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}", "", normalized)
        # Normalize whitespace before hashing to catch slight variations
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def parse_blocks(self, markdown: str) -> List[Block]:
        """Parses markdown text into semantic blocks while preserving code/tables."""
        blocks = []

        raw_lines = markdown.split("\n")

        current_block_lines = []
        in_code_block = False
        in_table = False

        for line in raw_lines:
            code_fences = line.count("```")

            if code_fences % 2 != 0:
                in_code_block = not in_code_block

            # Table detection: consecutive lines starting and ending with |
            # or containing |---|
            line_is_table_part = line.strip().startswith("|") and line.strip().endswith(
                "|"
            )
            if not in_code_block and not in_table and line_is_table_part:
                in_table = True
            elif not in_code_block and in_table and not line_is_table_part:
                in_table = False
                # Flush the table block
                content = "\n".join(current_block_lines).strip()
                if content:
                    blocks.append(self._create_block(content))
                current_block_lines = []

            current_block_lines.append(line)

            if not in_code_block and not in_table:
                # Flush normal line block
                content = "\n".join(current_block_lines).strip()
                if content:
                    blocks.append(self._create_block(content))
                current_block_lines = []

        # Handle trailing
        if current_block_lines:
            content = "\n".join(current_block_lines).strip()
            if content:
                blocks.append(self._create_block(content))

        # Assign position ratio
        total_blocks = len(blocks)
        for i, block in enumerate(blocks):
            block.metrics.position_ratio = i / total_blocks if total_blocks > 0 else 0.0

        return blocks

    def _create_block(self, content: str) -> Block:
        metrics = BlockMetrics()

        # Check type
        if content.startswith("```") and content.endswith("```"):
            metrics.is_code = True
        elif content.startswith("#"):
            metrics.is_heading = True
        elif "|---|" in content or "|---" in content or "---|" in content:
            metrics.is_table = True
        elif (
            content.startswith("- ")
            or content.startswith("* ")
            or re.match(r"^\d+\.\s", content)
        ):
            metrics.is_list = True

        words = re.findall(r"\b\w+\b", content.lower())
        metrics.word_count = len(words)

        if words:
            metrics.unique_word_ratio = len(set(words)) / len(words)
        else:
            metrics.unique_word_ratio = 0.0

        # Calculate link density
        # Matches [text](url)
        links = re.findall(r"\[([^\]]+)\]\([^\)]+\)", content)
        metrics.link_count = len(links)

        link_text_length = sum(len(text) for text in links)
        content_length = len(content)

        if content_length > 0:
            metrics.link_density = link_text_length / content_length

        return Block(
            content=content, content_hash=self._hash_content(content), metrics=metrics
        )

    def process_corpus_frequencies(self, all_documents_blocks: List[List[Block]]):
        """Pass 1: Count document frequency for all block hashes."""
        for doc_blocks in all_documents_blocks:
            # Use a set to only count a block once per document
            doc_hashes = set(b.content_hash for b in doc_blocks)
            for h in doc_hashes:
                self.block_document_counts[h] = self.block_document_counts.get(h, 0) + 1

    def clean_document_blocks(self, blocks: List[Block]) -> List[Block]:
        """Pass 2: Score and filter blocks for a single document."""
        cleaned = []

        for i, block in enumerate(blocks):
            # Update frequency metric
            df_count = self.block_document_counts.get(block.content_hash, 0)
            block.metrics.document_frequency = (
                df_count / self.total_documents if self.total_documents > 0 else 0.0
            )

            # 1. Protection rules (NEVER REMOVE)
            if (
                block.metrics.is_code
                or block.metrics.is_table
                or block.metrics.is_heading
            ):
                cleaned.append(block)
                continue

            # 2. Boilerplate Scoring
            score = 0.0
            reasons = []
            signals_triggered = []

            # Signal 1: Document Frequency
            if block.metrics.document_frequency > 0.80:
                score += 5.0
                signals_triggered.append("frequency_high")
                reasons.append(
                    f"High Frequency ({block.metrics.document_frequency:.0%})"
                )
            elif block.metrics.document_frequency > 0.40:
                score += 3.0
                signals_triggered.append("frequency_med")
                reasons.append(
                    f"Medium Frequency ({block.metrics.document_frequency:.0%})"
                )
            elif block.metrics.document_frequency > 0.10:
                score += 1.0
                signals_triggered.append("frequency_low")
                reasons.append(
                    f"Low Frequency ({block.metrics.document_frequency:.0%})"
                )

            # Signal 2: Position penalty (top 10% or bottom 10%)
            if block.metrics.position_ratio < 0.1 or block.metrics.position_ratio > 0.9:
                score += 1.5
                signals_triggered.append("edge_position")
                reasons.append(f"Edge Position ({block.metrics.position_ratio:.2f})")

            # Signal 3: Link density penalty
            if block.metrics.link_density > 0.8:
                score += 3.0
                signals_triggered.append("link_density_high")
                reasons.append(f"High Link Density ({block.metrics.link_density:.2f})")
            elif block.metrics.link_density > 0.5:
                score += 1.0
                signals_triggered.append("link_density_med")
                reasons.append(
                    f"Medium Link Density ({block.metrics.link_density:.2f})"
                )

            # Signal 4: Information density
            if block.metrics.word_count < 10 and block.metrics.link_count > 0:
                score += 2.0
                signals_triggered.append("info_density_low")
                reasons.append("Low Info Density (Short with Links)")
            elif block.metrics.word_count > 30 and block.metrics.link_density < 0.1:
                score -= 3.0
                reasons.append("High Info Density (Long Explanation)")

            # Signal 5: Content Diversity
            if block.metrics.word_count > 5 and block.metrics.unique_word_ratio < 0.5:
                score += 2.0
                signals_triggered.append("diversity_low")
                reasons.append(f"Low Diversity ({block.metrics.unique_word_ratio:.2f})")

            # Signal 6: Context (Check neighbors for high link density or edge position)
            # If previous or next block is a small link block
            context_penalty = 0.0
            if (
                i > 0
                and blocks[i - 1].metrics.link_density > 0.5
                and blocks[i - 1].metrics.word_count < 15
            ):
                context_penalty += 1.0
            if (
                i < len(blocks) - 1
                and blocks[i + 1].metrics.link_density > 0.5
                and blocks[i + 1].metrics.word_count < 15
            ):
                context_penalty += 1.0

            if context_penalty > 0:
                score += context_penalty
                signals_triggered.append("context_penalty")
                reasons.append(f"Context Penalty (+{context_penalty})")

            block.boilerplate_score = score
            block.triggered_signals = signals_triggered

            # Confidence Tiers for Removal
            is_removed = False

            # Tier 1: Extremely high frequency, short, low info (obvious chrome/nav)
            if (
                block.metrics.document_frequency > 0.95
                and block.metrics.word_count < 15
            ):
                is_removed = True
                reasons.append("Tier 1: Obvious Chrome")

            # Tier 2: Strong candidates (needs multiple signals)
            elif (
                score >= 6.0
                and len(set([s.split("_")[0] for s in signals_triggered])) >= 2
            ):
                # Requires score >= 6 and at least 2 different signal types (e.g. frequency + position)
                is_removed = True
                reasons.append("Tier 2: Multi-Signal Match")

            # Tier 3: Link walls
            elif (
                block.metrics.link_density > 0.9
                and block.metrics.document_frequency > 0.2
            ):
                is_removed = True
                reasons.append("Tier 3: Link Wall")

            if (
                is_removed and score > -1.0
            ):  # Prevent removal of strongly rewarded blocks
                block.is_removed = True
                block.removal_reason = " | ".join(reasons)
            else:
                cleaned.append(block)

        return cleaned
