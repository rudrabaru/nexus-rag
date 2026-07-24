from typing import List, Dict
from src.processing.models import Block
from src.processing.block_parser import BlockParser

class DocumentCleaner:
    def __init__(self, total_documents: int):
        self.total_documents = total_documents
        self.block_document_counts: Dict[str, int] = {}

    def parse_blocks(self, markdown: str) -> List[Block]:
        return BlockParser.parse_blocks(markdown)

    def process_corpus_frequencies(self, all_documents_blocks: List[List[Block]]):
        """Pass 1: Count document frequency for all block hashes."""
        for doc_blocks in all_documents_blocks:
            doc_hashes = set(b.content_hash for b in doc_blocks)
            for h in doc_hashes:
                self.block_document_counts[h] = self.block_document_counts.get(h, 0) + 1

    def clean_document_blocks(self, blocks: List[Block]) -> List[Block]:
        """Pass 2: Score and filter blocks for a single document."""
        cleaned = []

        for i, block in enumerate(blocks):
            df_count = self.block_document_counts.get(block.content_hash, 0)
            block.metrics.document_frequency = (
                df_count / self.total_documents if self.total_documents > 0 else 0.0
            )

            if (
                block.metrics.is_code
                or block.metrics.is_table
                or block.metrics.is_heading
            ):
                cleaned.append(block)
                continue

            score = 0.0
            reasons = []
            signals_triggered = []

            if block.metrics.document_frequency > 0.80:
                score += 5.0
                signals_triggered.append("frequency_high")
                reasons.append(f"High Frequency ({block.metrics.document_frequency:.0%})")
            elif block.metrics.document_frequency > 0.40:
                score += 3.0
                signals_triggered.append("frequency_med")
                reasons.append(f"Medium Frequency ({block.metrics.document_frequency:.0%})")
            elif block.metrics.document_frequency > 0.10:
                score += 1.0
                signals_triggered.append("frequency_low")
                reasons.append(f"Low Frequency ({block.metrics.document_frequency:.0%})")

            if block.metrics.position_ratio < 0.1 or block.metrics.position_ratio > 0.9:
                score += 1.5
                signals_triggered.append("edge_position")
                reasons.append(f"Edge Position ({block.metrics.position_ratio:.2f})")

            if block.metrics.link_density > 0.8:
                score += 3.0
                signals_triggered.append("link_density_high")
                reasons.append(f"High Link Density ({block.metrics.link_density:.2f})")
            elif block.metrics.link_density > 0.5:
                score += 1.0
                signals_triggered.append("link_density_med")
                reasons.append(f"Medium Link Density ({block.metrics.link_density:.2f})")

            if block.metrics.word_count < 10 and block.metrics.link_count > 0:
                score += 2.0
                signals_triggered.append("info_density_low")
                reasons.append("Low Info Density (Short with Links)")
            elif block.metrics.word_count > 30 and block.metrics.link_density < 0.1:
                score -= 3.0
                reasons.append("High Info Density (Long Explanation)")

            if block.metrics.word_count > 5 and block.metrics.unique_word_ratio < 0.5:
                score += 2.0
                signals_triggered.append("diversity_low")
                reasons.append(f"Low Diversity ({block.metrics.unique_word_ratio:.2f})")

            context_penalty = 0.0
            if i > 0 and blocks[i - 1].metrics.link_density > 0.5 and blocks[i - 1].metrics.word_count < 15:
                context_penalty += 1.0
            if i < len(blocks) - 1 and blocks[i + 1].metrics.link_density > 0.5 and blocks[i + 1].metrics.word_count < 15:
                context_penalty += 1.0

            if context_penalty > 0:
                score += context_penalty
                signals_triggered.append("context_penalty")
                reasons.append(f"Context Penalty (+{context_penalty})")

            block.boilerplate_score = score
            block.triggered_signals = signals_triggered
            is_removed = False

            if block.metrics.document_frequency > 0.95 and block.metrics.word_count < 15:
                is_removed = True
                reasons.append("Tier 1: Obvious Chrome")
            elif score >= 6.0 and len(set([s.split("_")[0] for s in signals_triggered])) >= 2:
                is_removed = True
                reasons.append("Tier 2: Multi-Signal Match")
            elif block.metrics.link_density > 0.9 and block.metrics.document_frequency > 0.2:
                is_removed = True
                reasons.append("Tier 3: Link Wall")

            if is_removed and score > -1.0:
                block.is_removed = True
                block.removal_reason = " | ".join(reasons)
            else:
                cleaned.append(block)

        return cleaned
