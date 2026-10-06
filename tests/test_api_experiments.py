"""Read access to experiments and test sets: scoped to the caller's workspace, and another workspace's rows do not exist."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)

REPORT = {
    "experiment": {"experiment_id": "e1", "name": "n", "tenant_id": "t1", "status": "complete"},
    "queries": 40, "synthetic": True, "relevance": "chunk", "baseline": "dense", "alpha": 0.05,
    "primary_metric": "mrr", "min_valid": 0.9, "comparison_top_k": 5,
    "trials": [{"label": "dense", "metrics": {"mrr": {"mean": 0.7}}}],
    "comparisons": [{"metric": "mrr", "candidate": "hybrid", "verdict": "insufficient evidence"}],
}


@pytest.fixture
def wired(app_state, monkeypatch):
    app_state.engine = object()
    app_state.testsets = MagicMock()
    return app_state


def test_every_route_needs_a_key(client, wired):
    for path in ("/v1/experiments", "/v1/experiments/e1", "/v1/test-sets", "/v1/test-sets/first"):
        assert client.get(path).status_code == 401, path


def test_the_list_shows_the_workspaces_experiments(client, wired, tenant_key, monkeypatch):
    seen = []
    row = {"experiment_id": "e1", "name": "dense vs hybrid", "dataset_name": "rust", "status": "complete",
           "created_at": NOW, "finished_at": NOW, "tenant_id": "t1"}
    monkeypatch.setattr("src.api.routes.experiments.store.list_experiments", lambda engine, tenant, limit: seen.append(tenant) or [row])
    response = client.get("/v1/experiments", headers={"X-API-Key": tenant_key("t1")})
    assert response.status_code == 200 and seen == ["t1"]
    assert response.json()[0]["experiment_id"] == "e1" and "tenant_id" not in response.json()[0]


def test_an_experiment_of_another_workspace_is_reported_as_missing(client, wired, tenant_key, monkeypatch):
    monkeypatch.setattr("src.api.routes.experiments.store.get_experiment", lambda engine, eid: {"tenant_id": "someone-else"})
    monkeypatch.setattr("src.api.routes.experiments.build_report", lambda engine, eid: pytest.fail("the report must not be built"))
    assert client.get("/v1/experiments/e1", headers={"X-API-Key": tenant_key("t1")}).status_code == 404


def test_a_missing_experiment_is_the_same_404(client, wired, tenant_key, monkeypatch):
    monkeypatch.setattr("src.api.routes.experiments.store.get_experiment", lambda engine, eid: None)
    assert client.get("/v1/experiments/nope", headers={"X-API-Key": tenant_key("t1")}).status_code == 404


def test_an_own_experiment_returns_its_report(client, wired, tenant_key, monkeypatch):
    monkeypatch.setattr("src.api.routes.experiments.store.get_experiment", lambda engine, eid: {"tenant_id": "t1"})
    monkeypatch.setattr("src.api.routes.experiments.build_report", lambda engine, eid: REPORT)
    body = client.get("/v1/experiments/e1", headers={"X-API-Key": tenant_key("t1")}).json()
    assert body["synthetic"] is True and body["comparisons"][0]["verdict"] == "insufficient evidence"


def test_the_test_set_list_is_the_workspaces_own(client, wired, tenant_key):
    wired.testsets.list_sets.return_value = [
        {"name": "first", "status": "frozen", "content_hash": "abc", "created_at": NOW, "questions": {"accepted": 12, "rejected": 3}},
    ]
    response = client.get("/v1/test-sets", headers={"X-API-Key": tenant_key("t1")})
    wired.testsets.list_sets.assert_called_once_with("t1")
    assert response.json()[0]["questions"] == {"accepted": 12, "rejected": 3}


def test_a_test_set_detail_lists_its_questions_in_order(client, wired, tenant_key):
    wired.testsets.find.return_value = {"test_set_id": "id1", "name": "first", "status": "draft", "content_hash": None}
    question = {"query": "Why?", "reference_answer": "Because.", "difficulty": "hard", "category": "prose",
                "review_status": "pending", "lexical_overlap": 0.2, "source_chunk_ids": ["c1"], "source_text": "passage"}
    wired.testsets.load.return_value = {"meta": {"model": "groq/x"}, "abstained": ["c9"], "questions": [question, dict(question, query="How?")]}
    body = client.get("/v1/test-sets/first", headers={"X-API-Key": tenant_key("t1")}).json()
    wired.testsets.find.assert_called_once_with("t1", "first")
    assert [q["position"] for q in body["questions"]] == [0, 1] and body["abstained"] == 1
    assert body["questions"][0]["source_text"] == "passage"


def test_an_unknown_test_set_is_a_404(client, wired, tenant_key):
    wired.testsets.find.return_value = None
    assert client.get("/v1/test-sets/nope", headers={"X-API-Key": tenant_key("t1")}).status_code == 404
