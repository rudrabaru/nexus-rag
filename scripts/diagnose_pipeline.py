import sys
import os
import argparse
import json
import hashlib
import pickle
import re
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", default=None)
parser.add_argument("--query", default="tell me the core idea about Pulse project")
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()

D = "\n" + "="*70

# STEP 1: AUTH
print(D)
print("STEP 1 - AUTH: resolving tenant_id from API key")
from src.registry.database import DocumentRegistry
registry = DocumentRegistry()
DEMO_KEY = os.environ.get("DEMO_API_KEY", "")

if not args.api_key:
    print("  [INFO] No API key provided -> tenant_id = None (public-only mode)")
    tenant_id = None
elif args.api_key == DEMO_KEY:
    print("  [INFO] Key matches DEMO_API_KEY -> tenant_id = demo_tenant")
    tenant_id = "demo_tenant"
else:
    key_hash = hashlib.sha256(args.api_key.encode()).hexdigest()
    print(f"  Key hash (sha256 prefix): {key_hash[:16]}...")
    tenant_id = registry.validate_api_key(args.api_key)
    if tenant_id:
        print(f"  [OK]  Key VALID -> tenant_id = {tenant_id}")
    else:
        print("  [FAIL] Key NOT found in local SQLite DB.")
        print("         NOTE: The HF Space has its own separate SQLite DB.")
        print("         A key generated inside HF Space will NOT be valid locally.")
        tenant_id = None

# STEP 2: REGISTRY
print(D)
print("STEP 2 - REGISTRY: listing all documents in SQLite")
docs = registry.list_documents()
if not docs:
    print("  [FAIL] No documents found. DB path may differ from the running server, or ingestion failed.")
else:
    for d in docs:
        print(f"  source={d['source'][:60]}")
        print(f"    visibility={d['visibility']}  tenant_id={d['tenant_id']}  status={d['status']}  chunks={len(d['chunk_ids'])}")

# STEP 3: CHROMADB INSPECTION
print(D)
print("STEP 3 - CHROMADB: inspecting stored chunks and metadata")
from src.retrieving.vector_store import ChromaDBManager
vector_store = ChromaDBManager()
total = vector_store.get_collection_size()
print(f"  Total chunks in ChromaDB: {total}")

if total == 0:
    print("  [FAIL] Collection is EMPTY. All queries will return 0 results.")
else:
    sample = vector_store.collection.get(limit=30, include=["metadatas"])
    vis_counts = {}
    tid_counts = {}
    for meta in sample["metadatas"]:
        v = meta.get("visibility", "MISSING")
        t = meta.get("tenant_id", "NOT_SET")
        vis_counts[v] = vis_counts.get(v, 0) + 1
        tid_counts[t] = tid_counts.get(t, 0) + 1
    print(f"  Visibility distribution (sample {len(sample['ids'])}): {vis_counts}")
    print(f"  Tenant ID  distribution (sample {len(sample['ids'])}): {tid_counts}")
    print("\n  First 3 chunks:")
    for i, (cid, meta) in enumerate(zip(sample["ids"][:3], sample["metadatas"][:3])):
        print(f"    [{i}] id={cid[:14]}")
        print(f"         source_document = {meta.get('source_document','')[:55]}")
        print(f"         visibility      = {meta.get('visibility', 'MISSING')}")
        print(f"         tenant_id       = {meta.get('tenant_id', 'NOT_SET')}")

# STEP 4: FILTER SIMULATION
print(D)
print("STEP 4 - FILTER: testing ChromaDB where clauses directly")
if tenant_id:
    where_clause = {"$or": [{"visibility": {"$eq": "public"}}, {"tenant_id": {"$eq": tenant_id}}]}
    print(f"  tenant_id={tenant_id} -> $or compound filter (requires chromadb>=0.5.15)")
else:
    where_clause = {"visibility": {"$eq": "public"}}
    print("  tenant_id=None -> simple equality filter")
print(f"  where_clause = {json.dumps(where_clause)}")

if total > 0:
    try:
        r = vector_store.collection.get(where={"visibility": {"$eq": "public"}}, limit=5, include=["metadatas"])
        print(f"\n  [get] visibility=public    -> {len(r['ids'])} chunks")
    except Exception as e:
        print(f"\n  [get] visibility=public    -> FAILED: {e}")

    if tenant_id:
        try:
            r2 = vector_store.collection.get(where={"tenant_id": {"$eq": tenant_id}}, limit=5, include=["metadatas"])
            print(f"  [get] tenant_id={tenant_id[:8]}... -> {len(r2['ids'])} chunks")
        except Exception as e:
            print(f"  [get] tenant_id filter     -> FAILED: {e}")
        try:
            r3 = vector_store.collection.get(where=where_clause, limit=5, include=["metadatas"])
            print(f"  [get] $or compound         -> {len(r3['ids'])} chunks")
        except Exception as e:
            print(f"  [get] $or compound         -> FAILED: {e}")
            print("         This is the ChromaDB $or bug. Fix: chromadb>=0.5.15 + redeploy.")

# STEP 5: LIVE RETRIEVAL
print(D)
print("STEP 5 - RETRIEVAL: DenseRetriever.retrieve()")
print(f"  query={args.query}")
print(f"  tenant_id={tenant_id}  top_k={args.top_k}")
from src.retrieving.retriever import DenseRetriever
from src.embedding.config import EmbeddingConfig
retriever = DenseRetriever(vector_store=vector_store, embedding_config=EmbeddingConfig())
result = retriever.retrieve(query=args.query, top_k=args.top_k, tenant_id=tenant_id)
print(f"\n  Embedding latency : {result.embedding_latency_ms:.1f}ms")
print(f"  Search latency    : {result.search_latency_ms:.1f}ms")
print(f"  Chunks returned   : {len(result.chunks)}")
if not result.chunks:
    print("  [FAIL] 0 chunks returned. Root cause confirmed in STEP 4 above.")
else:
    for i, c in enumerate(result.chunks):
        print(f"\n  [{i+1}] score={c.similarity_score:.4f}  vis={c.metadata.get('visibility')}  tenant={c.metadata.get('tenant_id','NOT_SET')}")
        print(f"       source={c.source_document[:55]}")
        print(f"       text[:120]={c.text[:120].strip()}")

# STEP 6: BM25
print(D)
print("STEP 6 - BM25: sparse index inspection")
bm25_path = Path(".chroma_db/bm25_index.pkl")
if not bm25_path.exists():
    print(f"  [FAIL] BM25 index not found at {bm25_path}")
else:
    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)
    chunks = bm25_data.get("chunks", [])
    print(f"  Chunks indexed: {len(chunks)}")
    vis_c = {}
    tid_c = {}
    for c in chunks:
        v = c.get("visibility", "MISSING")
        t = c.get("tenant_id", "NOT_SET")
        vis_c[v] = vis_c.get(v, 0) + 1
        tid_c[t] = tid_c.get(t, 0) + 1
    print(f"  Visibility : {vis_c}")
    print(f"  Tenant IDs : {tid_c}")
    import numpy as np
    q_clean = re.sub(r"[^\w\s]", " ", args.query.lower()).split()
    scores = bm25_data["bm25"].get_scores(q_clean)
    top = np.argsort(scores)[::-1][:5]
    print("\n  Top-5 BM25 matches:")
    for rank, idx in enumerate(top):
        ch = chunks[idx]
        print(f"    [{rank+1}] score={scores[idx]:.4f}  vis={ch.get('visibility')}  tenant={ch.get('tenant_id','?')}")
        print(f"         text={str(ch.get('chunk_text',''))[:80].strip()}")

print(D)
print("DIAGNOSTIC COMPLETE")
print(D)
