"""Query embeddings on real Postgres: stored losslessly, never overwritten, scoped to their index."""
from src.stores.query_embeddings import QueryEmbeddingStore, query_key


def test_a_vector_comes_back_as_it_went_in(pg_engine):
    store = QueryEmbeddingStore(pg_engine)
    vector = [0.1234567, -0.5, 0.25] + [0.0] * 1021
    store.put("voyage:voyage-4", query_key("why?"), vector, 9)
    loaded, tokens = store.get("voyage:voyage-4", query_key("why?"))
    assert tokens == 9 and len(loaded) == 1024
    assert all(abs(a - b) < 1e-6 for a, b in zip(loaded, vector))  # float4: what the provider returned, not halved


def test_the_first_vector_for_a_query_wins_and_other_indexes_are_separate(pg_engine):
    store = QueryEmbeddingStore(pg_engine)
    key = query_key("same question")
    store.put("a:m", key, [1.0], 3)
    store.put("a:m", key, [2.0], 4)  # a rerun racing the first: ignored, not an error
    assert store.get("a:m", key) == ([1.0], 3)
    assert store.get("b:m", key) is None
