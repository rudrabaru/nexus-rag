"""Moving a test set between its working form (a Draft) and Postgres, and in and out of dataset files."""
from typing import List

from src.evaluation.dataset import EvaluationQuery, load_dataset, write_dataset
from src.stores.testsets import ACCEPTED, QUESTION_FIELDS, TestSetError, TestSetStore
from src.testsets.models import Draft, DraftItem


def load_draft(store: TestSetStore, test_set_id: str) -> Draft:
    data = store.load(test_set_id)
    return Draft(meta=data["meta"], abstained=data["abstained"], items=[DraftItem(**q) for q in data["questions"]])


def save_draft(store: TestSetStore, test_set_id: str, draft: Draft) -> None:
    store.save(
        test_set_id, draft.meta, draft.abstained,
        [item.model_dump(include=set(QUESTION_FIELDS)) for item in draft.items],
    )


def accepted_queries(draft: Draft) -> List[EvaluationQuery]:
    return [item.to_query() for item in draft.items if item.review_status == ACCEPTED]


def import_dataset(store: TestSetStore, tenant_id: str, name: str, path: str) -> str:
    """Loads a dataset file as a frozen test set (every question in it counts as reviewed). Returns its content hash."""
    if store.find(tenant_id, name):
        raise TestSetError(f"Workspace {tenant_id!r} already has a test set named {name!r}.")
    dataset = load_dataset(path)
    draft = Draft(
        meta={"imported_from": dataset.name, "file_sha256": dataset.content_hash},
        items=[DraftItem(**q.model_dump(), review_status=ACCEPTED) for q in dataset.queries],
    )
    test_set_id = store.create(tenant_id, name, draft.meta)
    save_draft(store, test_set_id, draft)
    return store.freeze(test_set_id)


def export_dataset(store: TestSetStore, tenant_id: str, name: str, path: str) -> int:
    """Writes the accepted questions of a test set as a dataset file. Returns how many."""
    head = store.find(tenant_id, name)
    if head is None:
        raise TestSetError(f"Workspace {tenant_id!r} has no test set {name!r}.")
    queries = [EvaluationQuery(**q) for q in store.accepted_questions(head["test_set_id"])]
    write_dataset(path, queries)
    return len(queries)
