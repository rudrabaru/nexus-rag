"""Generator and retriever wiring, and checkpoints that let a retry resume."""
from types import SimpleNamespace
import httpx
import pytest
from src.config import get_settings
from src.embedding import embedder as embedder_module
from src.embedding.generator import EmbeddingGenerator, apportion, embedding_input, input_hash
from src.embedding.pacing import WINDOW_SECONDS
from src.embedding.providers import build_embedder
from src.retrieving.dense import DenseRetriever
from src.errors import EmbeddingRejectedError
from tests.support.embedding import VECTOR, voyage


def make_chunk(text="hello", heading_path=("Guide", "Setup")):
    from src.chunking.metadata import ChunkMetadata

    return ChunkMetadata(
        chunk_id="c0", source_url="u", source_document="Doc", title="T", heading_path=list(heading_path),
        chunk_text=text, token_count=1, 
        document_version="v", chunk_version="v", tenant_id="t", doc_id="d",
    )


class MemoryCheckpoints:
    def __init__(self):
        self.rows = {}

    def load(self, job_id):
        from src.stores.checkpoints import Checkpoint

        return {cid: Checkpoint(r["input_hash"], r["embedding"], r["tokens"]) for (j, cid), r in self.rows.items() if j == job_id}

    def save(self, job_id, rows):
        for row in rows:
            self.rows[(job_id, row["chunk_id"])] = row


def chunk_named(chunk_id, text="hello"):
    made = make_chunk(text=text)
    made.chunk_id = chunk_id
    return made


def two_per_request(monkeypatch):
    embedder = voyage(monkeypatch)
    embedder.max_texts_per_request = 2
    embedder.window = None  # pacing is not what is being tested
    return embedder


def test_the_embedding_input_carries_the_document_and_heading_path():
    assert embedding_input(make_chunk()) == "[Doc > Guide > Setup]\nhello"


def test_embedded_chunks_record_their_index(server, monkeypatch):
    embedder = voyage(monkeypatch)
    embedded, failed = EmbeddingGenerator(embedder).generate_embeddings([make_chunk()])
    assert failed == [] and embedded[0].index_id == "voyage:voyage-4" and embedded[0].embedding_model == "voyage-4"
    assert server.bodies[0]["input_type"] == "document"


def test_a_request_the_provider_rejects_for_good_stops_the_run_with_a_safe_message(server, monkeypatch):
    server.responses = [httpx.Response(401, json={"detail": "bad key sk-secret"})]
    generator = EmbeddingGenerator(voyage(monkeypatch))
    with pytest.raises(EmbeddingRejectedError) as exc:
        generator.generate_embeddings([make_chunk(), make_chunk(text="  ")])
    assert "401" in str(exc.value) and "sk-secret" not in str(exc.value)  # retrying cannot help; the key is not echoed
    assert len(server.bodies) == 1


def test_a_transient_failure_marks_the_chunks_failed_and_the_run_goes_on(server, monkeypatch):
    server.responses = [httpx.Response(500, json={"detail": "down"})] * 4
    embedder = voyage(monkeypatch)
    embedder.window = None  # pacing is not what is being tested
    generator = EmbeddingGenerator(embedder)
    embedded, failed = generator.generate_embeddings([make_chunk(), make_chunk(), make_chunk(text="  ")])
    assert embedded == [] and failed == [0, 1]  # the empty chunk is skipped, not failed
    assert "500" in generator.last_error


def test_a_429_waits_for_the_whole_window_not_a_few_seconds(server, monkeypatch):
    sleeps = []

    async def record(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(embedder_module.asyncio, "sleep", record)
    server.responses = [httpx.Response(429, json={"detail": "3 RPM"})]
    embedder = voyage(monkeypatch)
    embedder.window.reserve = lambda tokens, now=None: 0.0  # pacing is not what is being tested

    embedder.embed(["a"], "document")
    assert sleeps == [WINDOW_SECONDS]  # the old 2/4/8 second backoff spent every attempt inside one minute


def test_each_completed_request_is_checkpointed_with_its_share_of_the_providers_token_count(server, monkeypatch):
    store = MemoryCheckpoints()
    generator = EmbeddingGenerator(voyage(monkeypatch), store, "job-1")
    generator.generate_embeddings([chunk_named("a", "x" * 30), chunk_named("b", "y" * 10)])

    assert set(cid for _, cid in store.rows) == {"a", "b"}
    assert sum(r["tokens"] for r in store.rows.values()) == 7 == generator.provider_tokens


def test_a_retry_of_the_same_job_reuses_what_the_failed_attempt_finished(server, monkeypatch):
    store = MemoryCheckpoints()
    chunks = [chunk_named(name, name * 50) for name in "abcd"]
    ok = httpx.Response(200, json={"data": [{"embedding": VECTOR, "index": i} for i in range(2)], "usage": {"total_tokens": 7}})
    server.responses = [ok] + [httpx.Response(500, json={})] * 4  # the first request succeeds, the second never does
    first = EmbeddingGenerator(two_per_request(monkeypatch), store, "job-1")
    embedded, failed = first.generate_embeddings(chunks)
    assert [c.chunk_id for c in embedded] == ["a", "b"] and failed == [2, 3]

    server.bodies.clear()
    retry = EmbeddingGenerator(two_per_request(monkeypatch), store, "job-1")
    embedded, failed = retry.generate_embeddings(chunks)

    assert [c.chunk_id for c in embedded] == ["a", "b", "c", "d"] and failed == []
    assert len(server.bodies) == 1 and len(server.bodies[0]["input"]) == 2  # only the unfinished request was sent again


def test_a_checkpoint_is_never_reused_for_different_text_another_index_or_another_job(server, monkeypatch):
    store = MemoryCheckpoints()
    EmbeddingGenerator(voyage(monkeypatch), store, "job-1").generate_embeddings([chunk_named("a", "original")])

    server.bodies.clear()
    EmbeddingGenerator(voyage(monkeypatch), store, "job-1").generate_embeddings([chunk_named("a", "edited")])
    EmbeddingGenerator(voyage(monkeypatch), store, "job-2").generate_embeddings([chunk_named("a", "original")])
    ollama = build_embedder(get_settings(), "ollama:bge-m3")
    assert len(server.bodies) == 2  # the edited text and the other job were embedded afresh
    assert EmbeddingGenerator(ollama, store, "job-1")._saved["a"].input_hash != input_hash(ollama.index_id, "[Doc > Guide > Setup]\noriginal")


def test_tokens_are_apportioned_exactly():
    assert apportion(7, [30, 10]) == [6, 1] or sum(apportion(7, [30, 10])) == 7
    assert sum(apportion(1000, [1, 1, 1])) == 1000 and apportion(5, [0, 0]) == [5, 0]


def test_a_retriever_refuses_a_query_embedder_from_another_index(monkeypatch):
    store = SimpleNamespace(index_id="ollama:bge-m3")
    with pytest.raises(ValueError, match="not comparable"):
        DenseRetriever(store, voyage(monkeypatch))
