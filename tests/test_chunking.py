"""Chunking correctness: the verified defects (special tokens, tables beside headings, truncation) and the rules that replaced them."""
import pytest

from src.chunking.chunker import DocumentChunker
from src.chunking.metadata import ChunkingConfig
from src.chunking.parsers import parse_sections
from src.chunking.tokenizer import TokenCounter

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa "


def chunk(markdown, **config):
    chunker = DocumentChunker(ChunkingConfig(**config))
    return chunker.chunk_document({"url": "https://example.com/a", "title": "Doc", "markdown_content": markdown})


SMALL = dict(embedding_hard_limit=400, max_chunk_tokens=400, chunk_size=300, overlap=50, min_chunk_tokens=100)


def prose(words: int) -> str:
    return (WORDS * (words // 10 + 1)).strip()


# ── Token counting ───────────────────────────────────────────────────────────

def test_text_that_mentions_special_token_strings_is_still_counted():
    counter = TokenCounter()
    assert counter.count_tokens("The model stops at <|endoftext|> and resumes after <|fim_prefix|>.") > 10


# ── Tables and code beside headings ──────────────────────────────────────────

def test_a_table_directly_under_a_heading_does_not_corrupt_the_heading_path():
    markdown = "# Guide\n\n## Limits\n| a | b |\n|---|---|\n| 1 | 2 |\n## Third\nBody of the third section."
    sections = parse_sections(markdown)

    assert [s.heading_path for s in sections] == [["Guide"], ["Guide", "Limits"], ["Guide", "Third"]] or \
        [s.heading_path for s in sections][-2:] == [["Guide", "Limits"], ["Guide", "Third"]]
    limits = next(s for s in sections if s.title == "Limits")
    assert "| 1 | 2 |" in limits.text and "Body of the third" not in limits.text
    assert all("__TABLE" not in s.title and "\x00" not in s.title for s in sections)


def test_code_beside_headings_keeps_every_section_boundary():
    markdown = "## One\n```\ncode\n```\n## Two\ntext two"
    sections = parse_sections(markdown)
    assert [s.title for s in sections] == ["One", "Two"]
    assert "code" in sections[0].text and "text two" in sections[1].text


def test_a_placeholder_lookalike_in_the_document_is_left_alone():
    markdown = "## A\nThe literal __TABLE_BLOCK_0__ appears here.\n\n| x | y |\n|---|---|\n| 1 | 2 |\n"
    sections = parse_sections(markdown)
    assert "The literal __TABLE_BLOCK_0__ appears here." in sections[0].text and "| 1 | 2 |" in sections[0].text


def test_nul_bytes_never_reach_a_chunk():
    chunks = chunk("## Title\nBefore\x00after the byte, and more words to be a real chunk of text.")
    assert chunks and all("\x00" not in c.chunk_text for c in chunks)


# ── Oversized content is split, never cut ────────────────────────────────────

def test_a_block_over_the_embedding_limit_is_split_and_no_text_is_lost():
    lines = [f"row{i} {WORDS}" for i in range(400)]
    markdown = "## Big\n```\n" + "\n".join(lines) + "\n```"
    chunks = chunk(markdown, **SMALL)

    assert len(chunks) > 1
    assert all(c.token_count <= 400 for c in chunks) and not any("[TRUNCATED]" in c.chunk_text for c in chunks)
    joined = "\n".join(c.chunk_text for c in chunks)
    assert all(f"row{i} " in joined for i in range(400))
    assert all(c.heading_path == ["Big"] for c in chunks)


def test_a_table_split_across_chunks_repeats_its_header_row():
    rows = [f"| row{i} | {WORDS.strip()} |" for i in range(300)]
    markdown = "## Data\n| name | note |\n|---|---|\n" + "\n".join(rows)
    chunks = chunk(markdown, **SMALL)

    tables = [c for c in chunks if c.contains_table]
    assert len(tables) > 1 and all("| name | note |" in c.chunk_text for c in tables)
    assert all(c.token_count <= 400 for c in chunks)


def test_a_single_line_longer_than_the_limit_is_split_too():
    chunks = chunk("## Wall\n" + ("word " * 3000), **SMALL)
    assert len(chunks) > 1 and all(c.token_count <= 400 for c in chunks)


# ── Merging never crosses unrelated heading paths ────────────────────────────

def test_tiny_sibling_sections_stay_separate_chunks():
    markdown = "# Guide\n## Alpha\nShort alpha note.\n## Beta\nShort beta note."
    chunks = chunk(markdown)
    paths = [c.heading_path for c in chunks]
    assert ["Guide", "Alpha"] in paths and ["Guide", "Beta"] in paths
    for c in chunks:
        assert not ("alpha note" in c.chunk_text and "beta note" in c.chunk_text)


def test_a_heading_only_chunk_still_merges_into_its_own_body():
    chunks = chunk("# Guide\n## Alpha\nThe body that belongs to alpha and explains the topic in a sentence or two.")
    assert any(c.heading_path == ["Guide", "Alpha"] and "body that belongs" in c.chunk_text for c in chunks)


# ── A failing document is reported, not silently dropped ─────────────────────

def test_a_document_that_cannot_be_chunked_is_reported_with_its_reason(monkeypatch):
    chunker = DocumentChunker()
    original = chunker._split_content

    def explode(content, url, title, doc_name):
        if "bad" in url:
            raise ValueError("boom")
        return original(content, url, title, doc_name)

    monkeypatch.setattr(chunker, "_split_content", explode)
    docs = [
        {"url": "https://example.com/good", "title": "G", "markdown_content": "## A\n" + prose(40)},
        {"url": "https://example.com/bad", "title": "B", "markdown_content": "## A\n" + prose(40)},
    ]

    chunks = chunker.chunk_batch(docs)

    assert chunks and all("good" in c.source_url for c in chunks)
    assert [(f.url, "boom" in f.reason) for f in chunker.failures] == [("https://example.com/bad", True)]


@pytest.mark.parametrize("markdown", ["", "   \n\n  "])
def test_an_empty_document_yields_no_chunks(markdown):
    assert chunk(markdown) == []
