from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.crawling.metadata import CrawledDocument
from src.errors import UnprocessableSourceError
from src.ingestion.embedding_worker import EmbeddingUnavailableError, EmbeddingWorker
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
    """A chunker error must not be swallowed into a document that completes with 0 chunks."""
    monkeypatch.setattr("src.chunking.chunker.DocumentChunker._split_content", MagicMock(side_effect=RuntimeError("boom")))
    with pytest.raises(UnprocessableSourceError):
        process_documents([page(1)], "tenant-1", "doc-1", embedding_generator=embedding_generator())


def make_worker(chunk_count=3, embedded_indices=None, embedding_text="e"):
    """embedded_indices=None embeds every chunk; otherwise only those indices succeed."""
    input_chunks = [SimpleNamespace(chunk_id=f"c{i}", chunk_text=embedding_text, token_count=10) for i in range(chunk_count)]
    kept = range(chunk_count) if embedded_indices is None else embedded_indices
    embedded = [SimpleNamespace(chunk_id=f"c{i}", token_count=10) for i in kept]
    failed = sorted(set(range(chunk_count)) - set(kept))

    generator = MagicMock()
    generator.generate_embeddings.return_value = (embedded, failed)
    generator.last_error = None  # matches EmbeddingGenerator's real default
    return EmbeddingWorker(generator), input_chunks


def test_embedding_worker_reports_complete_when_nothing_fails():
    worker, chunks = make_worker()
    outcome = worker.embed(chunks)
    assert outcome.status == "complete"
    assert len(outcome.chunks) == 3
    assert outcome.failed_indices == []
    assert outcome.metadata is None


def test_embedding_worker_reports_partial_success_on_failed_batches():
    worker, chunks = make_worker(embedded_indices=[0, 2])
    outcome = worker.embed(chunks)
    assert outcome.status == "partial_success"
    assert outcome.failed_indices == [1]
    assert outcome.metadata["failed_chunk_indices"] == [1]


def test_embedding_worker_raises_when_nothing_could_be_embedded():
    """A total embedding failure (API outage, dead key) must not be reported as a silent success."""
    worker, chunks = make_worker(embedded_indices=[])
    with pytest.raises(EmbeddingUnavailableError):
        worker.embed(chunks)


def test_embedding_worker_reports_increasing_progress():
    chunks = [SimpleNamespace(chunk_id=f"c{i}", chunk_text="e", token_count=10) for i in range(120)]  # 3 batches of <=50
    generator = MagicMock()
    generator.generate_embeddings.side_effect = lambda batch: (batch, [])
    worker = EmbeddingWorker(generator)

    seen = []
    worker.embed(chunks, on_progress=seen.append)

    assert seen == sorted(seen) and len(seen) == 3
    assert seen[-1] == 99
