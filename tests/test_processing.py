"""Cleaning and fake-heading rules: removal needs structural evidence, and little evidence preserves the content."""
from src.crawling.readers import FetchedPage, ReaderError, _readable
from src.parsing.structure import promote_fake_headings
from src.processing.block_parser import BlockParser
from src.processing.cleaner import DocumentCleaner

import pytest

SHARED = "All rights are reserved by the publisher of this material"


def run_cleaner(documents):
    cleaner = DocumentCleaner(total_documents=len(documents))
    all_blocks = [BlockParser.parse_blocks(d) for d in documents]
    cleaner.process_corpus_frequencies(all_blocks)
    return [[b for b in cleaner.clean_document_blocks(blocks)] for blocks in all_blocks]


def texts(blocks):
    return [b.content for b in blocks]


def test_a_sentence_shared_by_two_documents_is_not_treated_as_boilerplate():
    kept = run_cleaner([f"# One\n\nUnique content of one.\n\n{SHARED}", f"# Two\n\nUnique content of two.\n\n{SHARED}"])
    assert all(SHARED in texts(blocks) for blocks in kept)


def test_a_short_line_repeated_across_many_documents_is_removed_as_chrome():
    documents = [f"# Page {i}\n\nOriginal body number {i} with its own explanation.\n\nSkip to main content" for i in range(8)]
    kept = run_cleaner(documents)
    assert all("Skip to main content" not in texts(blocks) for blocks in kept)
    assert all(any("Original body" in t for t in texts(blocks)) for blocks in kept)


def test_unique_content_is_never_removed():
    documents = [f"# Page {i}\n\nOriginal body number {i} with its own explanation." for i in range(8)]
    kept = run_cleaner(documents)
    assert all(len(blocks) == 2 for blocks in kept)


def test_bold_lines_become_headings_only_when_the_document_has_none():
    bold = "**Setup Guide**\n\nText.\n"
    assert "## Setup Guide" in promote_fake_headings(bold)
    real = "# Title\n\n" + bold
    assert promote_fake_headings(real) == real


def test_a_hash_line_inside_a_code_fence_is_not_a_real_heading():
    markdown = "```\n# a comment\n```\n\n**Setup Guide**\n\nText.\n"
    assert "## Setup Guide" in promote_fake_headings(markdown)


def page(markdown):
    return FetchedPage(url="https://example.com/a", title="T", markdown=markdown, provider="jina")


def test_a_short_page_is_kept_not_rejected_for_being_short():
    assert _readable(page("A short but real page with a dozen words, no more than that here.")).markdown


def test_a_page_with_no_words_is_unreadable():
    with pytest.raises(ReaderError):
        _readable(page("  \n\n  "))


def test_identical_text_has_one_content_key_whatever_its_whitespace():
    from src.jobs.fetch_tasks import content_key

    assert content_key("Access  denied\n\nplease log in") == content_key("Access denied please log in")
    assert content_key("Access denied") != content_key("Access granted")


def test_an_outcome_with_unchunked_documents_is_a_partial_success_that_names_them():
    from src.ingestion.embedding_worker import EmbeddingOutcome

    outcome = EmbeddingOutcome(chunks=[], failed_indices=[], total_chunks=3, error_reason="1 document(s) could not be chunked")
    assert outcome.status == "complete"
    outcome.unchunked_sources = ["https://example.com/bad"]
    assert outcome.status == "partial_success"
    assert outcome.metadata["unchunked_sources"] == ["https://example.com/bad"]
