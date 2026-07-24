import re
from typing import List

def normalize_identifier(text: str) -> List[str]:
    # Split by any non-alphanumeric separator
    tokens = re.split(r"[^a-z0-9]+", str(text).lower())
    return [t for t in tokens if t]

def is_sublist(sub: List[str], lst: List[str]) -> bool:
    # Checks if 'sub' is a continuous sublist of 'lst'
    if not sub:
        return True
    if not lst:
        return False
    n, m = len(sub), len(lst)
    for i in range(m - n + 1):
        if lst[i : i + n] == sub:
            return True
    return False

def evaluate_chunk(c, q, i, rank, exact_match_rank):
    import json
    from .evaluation_models import RetrievedChunkInfo

    doc_url = c.metadata.get("source_url", c.source_document)
    raw_path = c.metadata.get("heading_path", "")
    heading_path = []
    if raw_path:
        try:
            parsed = json.loads(raw_path)
            heading_path = parsed if isinstance(parsed, list) else [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            heading_path = [s.strip() for s in str(raw_path).split(" > ") if s.strip()]
            
    heading_path_str = " > ".join(heading_path) if isinstance(heading_path, list) else str(heading_path)
    section_title = c.metadata.get("section_title", "")
    raw_identifier = doc_url
    normalized_identifier = normalize_identifier(raw_identifier)

    doc_match = False
    matched_target = ""
    matching_rule_used = ""
    classification_reason = "No acceptable targets found in retrieved identifiers"

    if q.acceptable_documents:
        for acc in q.acceptable_documents:
            if acc in raw_identifier:
                doc_match = True
                matched_target = acc
                matching_rule_used = "Exact Substring Match"
                classification_reason = f"Raw identifier contains exact acceptable target: '{acc}'"
                break

        if not doc_match:
            for acc in q.acceptable_documents:
                norm_acc = normalize_identifier(acc)
                if is_sublist(norm_acc, normalized_identifier):
                    doc_match = True
                    matched_target = acc
                    matching_rule_used = "Normalized Token Sublist Match"
                    classification_reason = f"Normalized identifier contains normalized target tokens: {norm_acc}"
                    break

    heading_match = False
    if q.acceptable_headings:
        for acc_head in q.acceptable_headings:
            norm_acc_head = normalize_identifier(acc_head)
            norm_sec_title = normalize_identifier(section_title)
            norm_head_path = normalize_identifier(heading_path_str)

            if is_sublist(norm_acc_head, norm_sec_title) or is_sublist(norm_acc_head, norm_head_path):
                heading_match = True
                break

    match_type = "No Match" if not doc_match else "Partial Match"
    if doc_match:
        if q.acceptable_headings and heading_match:
            match_type = "Exact Match"
            classification_reason += " AND Heading matched exactly."
        elif not q.acceptable_headings:
            match_type = "Exact Match"
            classification_reason += " (No heading constraints)."
        else:
            match_type = "Partial Match"
            classification_reason += " BUT Heading constraints failed."

    chunk_info = RetrievedChunkInfo(
        chunk_id=c.chunk_id,
        source_document=c.source_document,
        similarity_score=c.similarity_score,
        text=c.text,
        section_title=section_title,
        heading_path=heading_path if isinstance(heading_path, list) else [],
        metadata=c.metadata,
        raw_identifier=raw_identifier,
        normalized_identifier=normalized_identifier,
        matched_target=matched_target,
        matching_rule_used=matching_rule_used,
        classification_reason=classification_reason,
        match_type=match_type,
    )

    if match_type in ["Exact Match", "Partial Match"] and rank == -1:
        rank = i + 1
    if match_type == "Exact Match" and exact_match_rank == -1:
        exact_match_rank = i + 1

    return chunk_info, doc_url, rank, exact_match_rank, match_type
