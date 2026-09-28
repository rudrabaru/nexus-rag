from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.crawling.metadata import CrawledDocument
from src.ingestion.errors import UnprocessableSourceError
from src.ingestion.pipeline import process_documents

NAV = "[Home](/home) [Docs](/docs) [Blog](/blog)"


def page(n: int) -> CrawledDocument:
    body = f"Topic {n} explains how component {n} is configured, deployed and monitored in production systems. " * 3
    return CrawledDocument(url=f"https://example.com/p{n}", title=f"Page {n}", markdown_content=f"{NAV}\n\n# Page {n}\n\n{body}")


def embedding_generator():
    generator = MagicMock()
    generator.embedder = SimpleNamespace(index_id="test:model")
    generator.generate_embeddings.side_effect = lambda batch: (batch, [])
    generator.last_error = None
    return generator


def test_a_block_repeated_across_every_page_of_a_job_is_not_indexed():
    events = MagicMock()
    outcome = process_documents(
        [page(1), page(2), page(3)], "tenant-1", "doc-1", pipeline_logger=events, embedding_generator=embedding_generator(),
    )

    assert outcome.chunks
    assert all("[Home]" not in c.chunk_text for c in outcome.chunks)
    assert all(c.tenant_id == "tenant-1" and c.doc_id == "doc-1" for c in outcome.chunks)
    audit = events.log_event.call_args.kwargs
    assert audit["blocks_removed"] == 3  # the audit counts what was actually dropped, one nav block per page


def test_a_single_page_keeps_its_links_because_frequency_needs_several_documents():
    outcome = process_documents([page(1)], "tenant-1", "doc-1", embedding_generator=embedding_generator())
    assert any("[Home]" in c.chunk_text for c in outcome.chunks)


def test_no_chunk_at_all_fails_the_job_instead_of_completing_it_empty(monkeypatch):
    """Regression: a chunker error was swallowed and the document completed with 0 chunks."""
    monkeypatch.setattr("src.chunking.chunker.DocumentChunker._split_content", MagicMock(side_effect=RuntimeError("boom")))
    with pytest.raises(UnprocessableSourceError):
        process_documents([page(1)], "tenant-1", "doc-1", embedding_generator=embedding_generator())
