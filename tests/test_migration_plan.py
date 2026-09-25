import json

import pytest

from scripts.migrate_legacy.plan import UNKNOWN_TENANT, build_plan

DIM = 4


def point(chunk_id, tenant="tenant-1", source="Doc A", **payload):
    return {
        "payload": {
            "chunk_id": chunk_id, "tenant_id": tenant, "source_document": source,
            "source_url": f"https://{source}", "heading_path": json.dumps(["H1"]), "document_text": f"text {chunk_id}",
            **payload,
        },
        "vector": [0.5] * DIM,
    }


def legacy(documents=(), jobs=(), api_keys=(), query_logs=()):
    return {"documents": list(documents), "jobs": list(jobs), "api_keys": list(api_keys), "query_logs": list(query_logs)}


def document(doc_id, tenant, chunk_ids, **extra):
    return {"doc_id": doc_id, "tenant_id": tenant, "source": f"src-{doc_id}", "format": "pdf", "status": "complete",
            "chunk_ids": list(chunk_ids), "stats": {}, "ingested_at": "2026-07-24T12:00:00+00:00", **extra}


def plan_for(points, registry):
    return build_plan(points, registry, "jina-embeddings-v3", DIM)


def test_chunks_are_owned_by_the_registry_document_that_lists_them():
    plan = plan_for([point("c1"), point("c2")], legacy([document("doc-1", "tenant-1", ["c1", "c2"])]))
    assert {c["doc_id"] for c in plan["chunks"]} == {"doc-1"}
    assert plan["report"]["chunks_in_reconstructed_documents"] == 0


def test_unclaimed_chunks_get_one_reconstructed_document_per_tenant_and_source():
    points = [point("c1", source="A"), point("c2", source="A"), point("c3", source="B"), point("c4", tenant="t2", source="A")]
    plan = plan_for(points, legacy())

    reconstructed = [d for d in plan["documents"] if d["format"] == "reconstructed"]
    assert len(reconstructed) == 3
    assert plan["report"]["chunks_in_reconstructed_documents"] == 4
    by_chunk = {c["chunk_id"]: c["doc_id"] for c in plan["chunks"]}
    assert by_chunk["c1"] == by_chunk["c2"] != by_chunk["c3"]
    assert by_chunk["c1"] != by_chunk["c4"]


def test_reconstructed_doc_ids_are_deterministic_so_reruns_converge():
    first = plan_for([point("c1")], legacy())["chunks"][0]["doc_id"]
    second = plan_for([point("c1")], legacy())["chunks"][0]["doc_id"]
    assert first == second and first.startswith("rc_")


def test_a_chunk_overwritten_by_another_tenant_is_reported_not_invented():
    """Qdrant keyed points by chunk_id alone, so tenant-2's ingestion replaced tenant-1's copy."""
    registry = legacy([document("doc-a", "tenant-1", ["c1"]), document("doc-b", "tenant-2", ["c1"])])
    plan = plan_for([point("c1", tenant="tenant-2")], registry)

    assert [(c["tenant_id"], c["doc_id"]) for c in plan["chunks"]] == [("tenant-2", "doc-b")]
    assert plan["report"]["registry_chunk_ids_missing_from_qdrant"] == 1


def test_heading_path_is_parsed_into_a_list_and_provenance_recorded():
    [chunk] = plan_for([point("c1")], legacy())["chunks"]
    assert chunk["heading_path"] == ["H1"]
    assert chunk["embedding_model"] == "jina-embeddings-v3"
    assert chunk["chunk_text"] == "text c1"


def test_missing_tenant_falls_back_to_an_unreachable_tenant():
    [chunk] = plan_for([point("c1", tenant=None)], legacy())["chunks"]
    assert chunk["tenant_id"] == UNKNOWN_TENANT


def test_wrong_dimension_aborts_the_migration():
    bad = point("c1")
    bad["vector"] = [0.1] * (DIM + 1)
    with pytest.raises(ValueError):
        plan_for([bad], legacy())


def test_tenant_token_total_is_the_max_across_its_keys_not_the_sum():
    keys = [
        {"key_hash": "h1", "tenant_id": "t", "created_at": None, "total_embedding_tokens": 12800},
        {"key_hash": "h2", "tenant_id": "t", "created_at": None, "total_embedding_tokens": 0},
    ]
    plan = plan_for([], legacy(api_keys=keys))
    assert plan["tenants"][0]["total_embedding_tokens"] == 12800
    assert len(plan["api_keys"]) == 2


def test_jobs_of_unknown_documents_are_dropped():
    jobs = [{"job_id": "j1", "doc_id": "doc-1", "status": "complete"}, {"job_id": "j2", "doc_id": "ghost", "status": "failed"}]
    plan = plan_for([], legacy([document("doc-1", "tenant-1", [])], jobs=jobs))
    assert [j["job_id"] for j in plan["jobs"]] == ["j1"]
