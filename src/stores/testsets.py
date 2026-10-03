"""
Test sets in Postgres: a draft while questions are generated and reviewed, frozen once an
experiment may use it.

A frozen set is immutable and carries a content hash of its accepted questions, so two
experiments ran the same questions exactly when their hashes match. The working copy is saved as
a whole (`save`): sets hold tens to hundreds of questions, so replacing the rows in one
transaction is atomic, simple, and cheap. A crash mid-review never costs the decisions already saved.
"""
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Engine

from src.db.rows import row_to_dict, utcnow
from src.db.schema import test_questions, test_sets

DRAFT, FROZEN = "draft", "frozen"
PENDING, ACCEPTED, REJECTED = "pending", "accepted", "rejected"  # a question's review status


class TestSetError(RuntimeError):
    """A test set cannot be changed, found or frozen as asked."""

    __test__ = False  # not a pytest test class


def content_hash(queries: List[Dict[str, Any]]) -> str:
    """Hash of the accepted questions in canonical form: the identity of what an experiment is scored on."""
    canonical = json.dumps(queries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


DATASET_FIELDS = (
    "query", "acceptable_documents", "acceptable_headings", "source_chunk_ids", "reference_answer", "difficulty",
    "category", "origin", "lexical_overlap",
)
QUESTION_FIELDS = (
    "query", "reference_answer", "acceptable_documents", "acceptable_headings", "source_chunk_ids", "difficulty",
    "category", "origin", "lexical_overlap", "source_text", "review_status",
)


class TestSetStore:
    __test__ = False  # not a pytest test class

    def __init__(self, engine: Engine):
        self._engine = engine

    def find(self, tenant_id: str, name: str) -> Optional[Dict[str, Any]]:
        stmt = select(test_sets).where(test_sets.c.tenant_id == tenant_id, test_sets.c.name == name)
        with self._engine.connect() as conn:
            return row_to_dict(conn.execute(stmt).first())

    def list_sets(self, tenant_id: str) -> List[Dict[str, Any]]:
        """One tenant's sets with their question counts by review status."""
        count = func.count(test_questions.c.position)
        stmt = (
            select(test_sets.c.name, test_sets.c.status, test_sets.c.content_hash, test_sets.c.created_at,
                   test_questions.c.review_status, count.label("n"))
            .select_from(test_sets.outerjoin(test_questions, test_questions.c.test_set_id == test_sets.c.test_set_id))
            .where(test_sets.c.tenant_id == tenant_id)
            .group_by(test_sets.c.test_set_id, test_questions.c.review_status)
            .order_by(test_sets.c.created_at.desc())
        )
        with self._engine.connect() as conn:
            rows = [row_to_dict(r) for r in conn.execute(stmt)]
        sets: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            entry = sets.setdefault(row["name"], {k: row[k] for k in ("name", "status", "content_hash", "created_at")} | {"questions": {}})
            if row["review_status"]:
                entry["questions"][row["review_status"]] = row["n"]
        return list(sets.values())

    def create(self, tenant_id: str, name: str, meta: Dict[str, Any]) -> str:
        test_set_id = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(insert(test_sets).values(
                test_set_id=test_set_id, tenant_id=tenant_id, name=name, status=DRAFT, meta=meta, abstained=[],
            ))
        return test_set_id

    def load(self, test_set_id: str) -> Dict[str, Any]:
        """{"meta", "abstained", "questions": [...]} with the questions in order."""
        with self._engine.connect() as conn:
            head = conn.execute(select(test_sets).where(test_sets.c.test_set_id == test_set_id)).mappings().first()
            if head is None:
                raise TestSetError(f"No test set {test_set_id}")
            rows = conn.execute(
                select(*[test_questions.c[f] for f in QUESTION_FIELDS])
                .where(test_questions.c.test_set_id == test_set_id).order_by(test_questions.c.position)
            ).mappings().all()
        return {"meta": head["meta"], "abstained": head["abstained"], "questions": [dict(r) for r in rows]}

    def save(self, test_set_id: str, meta: Dict[str, Any], abstained: List[str], questions: List[Dict[str, Any]]) -> None:
        """Replaces the set's questions and abstentions, atomically. A frozen set refuses."""
        with self._engine.begin() as conn:
            status = conn.execute(select(test_sets.c.status).where(test_sets.c.test_set_id == test_set_id)).scalar_one_or_none()
            if status is None:
                raise TestSetError(f"No test set {test_set_id}")
            if status == FROZEN:
                raise TestSetError("A frozen test set cannot be changed. Copy it to a new set to revise it.")
            conn.execute(delete(test_questions).where(test_questions.c.test_set_id == test_set_id))
            if questions:
                conn.execute(insert(test_questions), [{"test_set_id": test_set_id, "position": i, **q} for i, q in enumerate(questions)])
            conn.execute(update(test_sets).where(test_sets.c.test_set_id == test_set_id).values(abstained=abstained, meta=meta))

    def accepted_questions(self, test_set_id: str) -> List[Dict[str, Any]]:
        """The accepted questions as dataset entries (no source text or review status), in order."""
        stmt = (
            select(*[test_questions.c[f] for f in DATASET_FIELDS])
            .where(test_questions.c.test_set_id == test_set_id, test_questions.c.review_status == ACCEPTED)
            .order_by(test_questions.c.position)
        )
        with self._engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings()]

    def freeze(self, test_set_id: str) -> str:
        """Makes the set immutable and returns the hash of its accepted questions."""
        accepted = self.accepted_questions(test_set_id)
        if not accepted:
            raise TestSetError("There are no accepted questions to freeze: review the draft first.")
        digest = content_hash(accepted)
        with self._engine.begin() as conn:
            conn.execute(
                update(test_sets).where(test_sets.c.test_set_id == test_set_id, test_sets.c.status == DRAFT)
                .values(status=FROZEN, content_hash=digest, frozen_at=utcnow())
            )
        return digest

    def delete_set(self, test_set_id: str) -> bool:
        with self._engine.begin() as conn:
            return conn.execute(delete(test_sets).where(test_sets.c.test_set_id == test_set_id)).rowcount > 0
