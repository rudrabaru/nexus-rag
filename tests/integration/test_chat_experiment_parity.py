"""What an experiment measures is what chat serves: the same RetrievalConfig returns the same chunks through both."""
import pytest

from src.services.chat_config import chat_retrieval_config
from src.services.chat_models import ChatQuery
from src.services.chat_service import ChatService
from src.config import get_settings
from src.evaluation import store
from src.evaluation.dataset import Dataset, EvaluationQuery
from src.evaluation.engine import run_experiment
from src.evaluation.spec import ExperimentSpec
from tests.integration.helpers import Stores
from src.retrieving.chunk_writes import write_chunks
from src.retrieving.pipeline import RetrievalResources
from tests.support.postgres import AxisEmbedder, add_document, chunk, unit_vector

pytestmark = pytest.mark.usefixtures("clean_tables")


@pytest.fixture
def corpus(pg_engine, monkeypatch):
    registry = Stores(pg_engine)
    add_document(registry, doc_id="doc-1", tenant="demo")
    with pg_engine.begin() as conn:
        write_chunks(conn, [
            chunk("k1", tenant="demo", chunk_text="rotate the signing keys every ninety days", vector=unit_vector(0)),
            chunk("k2", tenant="demo", chunk_text="the quarterly audit of access logs", vector=unit_vector(0, 1)),
            chunk("k3", tenant="demo", chunk_text="keys are stored in a vault", vector=unit_vector(0, 2)),
            chunk("k4", tenant="demo", chunk_text="unrelated notes about lunch", vector=unit_vector(3)),
        ])
    monkeypatch.setattr("src.retrieving.pipeline.build_embedder", lambda settings, index_id=None: AxisEmbedder())


@pytest.mark.parametrize("strategy", ["dense", "sparse", "hybrid"])
async def test_chat_and_an_experiment_trial_return_identical_chunks(pg_engine, pg_async_engine, corpus, monkeypatch, strategy):
    monkeypatch.setenv("RETRIEVAL_STRATEGY", strategy)
    get_settings.cache_clear()
    try:
        resources = RetrievalResources(get_settings(), pg_engine, pg_async_engine)
        body = ChatQuery(query="rotate signing keys", top_k=3)
        config = chat_retrieval_config(get_settings(), None, body.top_k, body.use_reranker)
        assert config.strategy == strategy

        chat = (await ChatService(resources, generator=None, evaluator=None).prepare("demo", body)).retrieval

        spec = ExperimentSpec(name="parity", dataset="inline", tenant_id="demo", trials={"chat": config.model_dump(mode="json")})
        queries = [EvaluationQuery(query=body.query, acceptable_documents=["doc-1"])]
        experiment_id = store.create_experiment(pg_engine, spec, Dataset(name="inline", content_hash="h", queries=queries))
        assert await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None) == "complete"

        trial = store.trials_of(pg_engine, experiment_id)[0]
        measured = [r["chunk_id"] for r in store.runs_of(pg_engine, trial["trial_id"])[0]["retrieved"]]
        assert measured == [c.chunk_id for c in chat.chunks] and measured  # same chunks, same order, not empty
    finally:
        get_settings.cache_clear()
