"""Reviewing drafts, datasets and configuration."""
import json
import pytest
from src.config import Settings
from src.evaluation.dataset import load_dataset, write_dataset
from src.stores.testsets import ACCEPTED, PENDING, REJECTED
from src.testsets.models import Draft, DraftItem
from src.testsets.repository import accepted_queries
from src.testsets.review import review


def pending_draft(n=3) -> Draft:
    items = [DraftItem(query=f"q{i}?", reference_answer=f"a{i}", acceptable_documents=["d"], source_chunk_ids=[f"c{i}"],
                       source_text="the source says things", lexical_overlap=0.1) for i in range(n)]
    return Draft(items=items)


def scripted(*answers):
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def test_review_applies_each_decision_and_saves_after_every_one():
    draft, saved = pending_draft(4), []
    review(draft, lambda d: saved.append([i.review_status for i in d.items]), ask=scripted("a", "r", "s", "q"), show=lambda _: None)
    assert [i.review_status for i in draft.items] == [ACCEPTED, REJECTED, PENDING, PENDING]
    assert len(saved) == 2  # skip and quit change nothing


def test_editing_replaces_the_text_recomputes_overlap_and_accepts():
    draft = pending_draft(1)
    review(draft, lambda d: None, ask=scripted("e", "the source says", ""), show=lambda _: None)
    item = draft.items[0]
    assert item.query == "the source says" and item.reference_answer == "a0"  # an empty reply keeps the answer
    assert item.review_status == ACCEPTED
    assert item.lexical_overlap == 1.0  # was 0.1 before the edit


def test_review_asks_again_after_an_unknown_choice_and_only_reviews_pending_items():
    draft = pending_draft(2)
    draft.items[0].review_status = REJECTED
    shown = []
    review(draft, lambda d: None, ask=scripted("what?", "a"), show=shown.append)
    assert [i.review_status for i in draft.items] == [REJECTED, ACCEPTED]
    assert "Choose a, e, r, s or q." in shown


def test_only_accepted_questions_become_a_dataset_the_engine_loads_in_chunk_mode(tmp_path):
    draft = pending_draft(3)
    draft.items[0].review_status, draft.items[2].review_status = ACCEPTED, REJECTED
    queries = accepted_queries(draft)
    assert [q.query for q in queries] == ["q0?"]

    path = str(tmp_path / "set.json")
    write_dataset(path, queries)
    loaded = load_dataset(path, relevance="chunk")
    assert loaded.queries[0].source_chunk_ids == ["c0"] and "source_text" not in json.loads(open(path, encoding="utf-8").read())[0]
    assert len(loaded.content_hash) == 64


def test_an_empty_dataset_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no queries"):
        write_dataset(str(tmp_path / "set.json"), [])


def test_llm_keys_are_checked_per_provider():
    settings = Settings(_env_file=None, gemini_api_key="g", groq_api_key="")
    assert settings.has_llm_key("gemini") and not settings.has_llm_key("groq") and settings.has_llm_key("some-local-provider")
