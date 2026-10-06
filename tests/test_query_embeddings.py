"""Saved query embeddings: an experiment's rerun does not spend the provider's rate limit again, and chat never stores a question."""
from src.embedding.embedder import EmbeddingBatch
from src.retrieving.dense import DenseRetriever
from src.stores.query_embeddings import query_key


class FakeEmbedder:
    index_id = "voyage:voyage-4"

    def __init__(self):
        self.calls = []

    async def aembed(self, texts, input_type):
        self.calls.append((texts, input_type))
        return EmbeddingBatch(vectors=[[0.5, 0.25]], tokens=7)

    def cost_usd(self, tokens):
        return tokens * 1e-6


class FakeChunkStore:
    index_id = "voyage:voyage-4"
    distance_metric = "cosine"

    async def search_dense(self, query_embedding, top_k, tenant_id):
        return []


class MemoryStore:
    def __init__(self, fail_put=False):
        self.rows, self.fail_put = {}, fail_put

    def get(self, index_id, key):
        return self.rows.get((index_id, key))

    def put(self, index_id, key, vector, tokens):
        if self.fail_put:
            raise RuntimeError("database down")
        self.rows[(index_id, key)] = (vector, tokens)


def retriever(embedder, saved=None):
    return DenseRetriever(FakeChunkStore(), embedder, saved)


async def test_a_cold_process_reuses_the_saved_vector_and_keeps_the_token_count():
    saved = MemoryStore()
    first = FakeEmbedder()
    await retriever(first, saved).retrieve("What is ownership?", tenant_id="t")
    assert len(first.calls) == 1 and len(saved.rows) == 1

    second = FakeEmbedder()  # a new process: empty memory cache, the same database
    result = await retriever(second, saved).retrieve("What is ownership?", tenant_id="t")
    assert second.calls == []
    assert result.embedding_tokens == 0  # nothing was spent this time
    assert result.query_embedding_tokens == 7  # what it costs, so a configuration is still charged fairly


async def test_the_saved_key_ignores_case_and_surrounding_space():
    saved = MemoryStore()
    await retriever(FakeEmbedder(), saved).retrieve("What is Ownership? ", tenant_id="t")
    again = FakeEmbedder()
    await retriever(again, saved).retrieve("  what is ownership?", tenant_id="t")
    assert again.calls == [] and query_key("A b") == query_key(" a B ")


async def test_without_a_store_nothing_is_kept_and_the_provider_is_asked_each_cold_start():
    for _ in range(2):
        embedder = FakeEmbedder()
        await retriever(embedder).retrieve("a chat question", tenant_id="t")
        assert len(embedder.calls) == 1


async def test_a_failure_to_save_does_not_fail_the_query():
    embedder = FakeEmbedder()
    result = await retriever(embedder, MemoryStore(fail_put=True)).retrieve("q", tenant_id="t")
    assert result.embedding_tokens == 7


def test_only_experiments_keep_embeddings():
    import inspect

    from src.retrieving.pipeline import RetrievalResources

    default = inspect.signature(RetrievalResources.__init__).parameters["keep_query_embeddings"].default
    assert default is False  # chat builds its resources with the default; only the evaluation CLI asks to keep
