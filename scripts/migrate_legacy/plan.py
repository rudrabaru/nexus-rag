"""
Pure transformation from legacy rows to Postgres rows. No I/O, so it is unit-tested directly.

Chunk ownership: Qdrant payloads carry no doc_id. A chunk's document is found through the
legacy registry's documents.chunk_ids. Chunks no document claims (their registry rows were
lost when an ephemeral SQLite disk was wiped) get one reconstructed document per
(tenant_id, source_document), marked format="reconstructed" so they stay identifiable.
"""
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from scripts.migrate_legacy.read_legacy import parse_json, parse_timestamp
from src.registry.rows import utcnow

UNKNOWN_TENANT = "unknown"  # the legacy payload default; no API key maps to it


def _reconstructed_doc_id(tenant_id: str, source_document: str) -> str:
    digest = hashlib.blake2b(f"{tenant_id}\0{source_document}".encode(), digest_size=16).hexdigest()
    return f"rc_{digest}"


def _heading_path(raw) -> List[str]:
    parsed = parse_json(raw, []) if isinstance(raw, str) else raw
    return [str(h) for h in parsed] if isinstance(parsed, list) else []


def build_plan(points: List[Dict[str, Any]], legacy: Dict[str, List[Dict[str, Any]]], embedding_model: str, dimension: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {"skipped_documents_without_tenant": 0}

    documents = {}
    owner: Dict[Tuple[str, str], str] = {}
    for d in legacy["documents"]:
        if not d.get("tenant_id"):
            report["skipped_documents_without_tenant"] += 1
            continue
        documents[d["doc_id"]] = {
            "doc_id": d["doc_id"],
            "tenant_id": d["tenant_id"],
            "source": d.get("source") or d["doc_id"],
            "format": d.get("format") or "unknown",
            "status": d.get("status") or "unknown",
            "visibility": d.get("visibility") or "private",
            "content_hash": d.get("content_hash"),
            "stats": d.get("stats") or {},
            "error": d.get("error"),
            "ingested_at": parse_timestamp(d.get("ingested_at")) or utcnow(),
            "updated_at": parse_timestamp(d.get("updated_at")) or utcnow(),
        }
        for chunk_id in d["chunk_ids"]:
            owner[(d["tenant_id"], chunk_id)] = d["doc_id"]

    chunks, orphans_by_doc = [], defaultdict(int)
    for point in points:
        payload, vector = point["payload"], point["vector"]
        if len(vector) != dimension:
            raise ValueError(f"Chunk {payload.get('chunk_id')} has {len(vector)} dimensions, expected {dimension}.")
        tenant_id = payload.get("tenant_id") or UNKNOWN_TENANT
        chunk_id = payload["chunk_id"]
        source_document = payload.get("source_document") or payload.get("source_url") or chunk_id

        doc_id = owner.get((tenant_id, chunk_id))
        if doc_id is None:
            doc_id = _reconstructed_doc_id(tenant_id, source_document)
            orphans_by_doc[doc_id] += 1
            documents.setdefault(doc_id, {
                "doc_id": doc_id,
                "tenant_id": tenant_id,
                "source": payload.get("source_url") or source_document,
                "format": "reconstructed",
                "status": "complete",
                "visibility": "private",
                "content_hash": None,
                "stats": {"reconstructed_from": "qdrant"},
                "error": None,
                "ingested_at": utcnow(),
                "updated_at": utcnow(),
            })

        chunks.append({
            "tenant_id": tenant_id,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_document": source_document,
            "source_url": payload.get("source_url"),
            "title": payload.get("title"),
            "section_title": payload.get("section_title"),
            "heading_path": _heading_path(payload.get("heading_path")),
            "content_type": payload.get("content_type"),
            "contains_code": bool(payload.get("contains_code")),
            "contains_table": bool(payload.get("contains_table")),
            "chunk_version": payload.get("chunk_version"),
            "document_version": payload.get("document_version"),
            "token_count": None,
            "chunk_text": payload.get("document_text") or "",
            "embedding": vector,
            "embedding_model": embedding_model,
        })

    jobs = [
        {
            "job_id": j["job_id"],
            "doc_id": j["doc_id"],
            "status": j.get("status") or "unknown",
            "progress_pct": j.get("progress_pct") or 0,
            "created_at": parse_timestamp(j.get("created_at")) or utcnow(),
            "finished_at": parse_timestamp(j.get("finished_at")),
            "error": j.get("error"),
            "metadata": j.get("metadata"),
        }
        for j in legacy["jobs"]
        if j.get("doc_id") in documents
    ]

    # The legacy counter was incremented on every key of a tenant, so each key holds the same
    # running total (or 0 for keys created later). The tenant total is the maximum, not the sum.
    tenant_tokens: Dict[str, int] = defaultdict(int)
    api_keys = []
    for k in legacy["api_keys"]:
        tenant_tokens[k["tenant_id"]] = max(tenant_tokens[k["tenant_id"]], k.get("total_embedding_tokens") or 0)
        api_keys.append({
            "key_hash": k["key_hash"],
            "tenant_id": k["tenant_id"],
            "key_prefix": None,
            "created_at": parse_timestamp(k.get("created_at")) or utcnow(),
            "revoked_at": None,
        })
    tenants = [
        {"tenant_id": t, "created_at": utcnow(), "total_embedding_tokens": tokens} for t, tokens in tenant_tokens.items()
    ]

    query_logs = []
    for q in legacy["query_logs"]:
        if not q.get("tenant_id"):
            continue
        row = {k: q.get(k) for k in (
            "log_id", "tenant_id", "query", "latency_ms", "tokens_used", "faithfulness_score", "provider",
            "embedding_tokens", "embedding_cost_usd", "generation_input_tokens", "generation_output_tokens",
            "generation_cost_usd", "rerank_cost_usd", "total_cost_usd",
        )}
        row["timestamp"] = parse_timestamp(q.get("timestamp")) or utcnow()
        row["details"] = json.loads(json.dumps(q.get("details") or {}, default=str))
        row["query"] = row["query"] or ""
        query_logs.append(row)

    present = {(c["tenant_id"], c["chunk_id"]) for c in chunks}
    report.update(
        qdrant_points=len(points),
        chunks_owned_by_registry_documents=len(chunks) - sum(orphans_by_doc.values()),
        chunks_in_reconstructed_documents=sum(orphans_by_doc.values()),
        reconstructed_documents=len(orphans_by_doc),
        registry_chunk_ids_missing_from_qdrant=sum(1 for key in owner if key not in present),
        documents=len(documents),
        jobs=len(jobs),
        api_keys=len(api_keys),
        query_logs=len(query_logs),
    )
    return {
        "tenants": tenants, "documents": list(documents.values()), "jobs": jobs, "chunks": chunks,
        "api_keys": api_keys, "query_logs": query_logs, "report": report,
    }
