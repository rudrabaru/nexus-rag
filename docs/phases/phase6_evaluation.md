# Phase 6: Evaluation

## Overview
Evaluation answers one question with evidence: is configuration B better than configuration A on these queries, or is the difference noise? An **experiment** runs one or more retrieval configurations (**trials**) over a frozen query set, stores every per-query result (**run**) in Postgres, and compares each trial with a baseline under a significance test. Generation and faithfulness judging are an optional second stage.

The engine is `src/evaluation/`; it is run from the command line. Starting experiments through the API and job queue comes with configuration search (item 13).

## Core Implementation Logic

### Datasets
A dataset is a JSON list of queries. Each names the documents that would answer it (`acceptable_documents`) and optionally the headings (`acceptable_headings`), plus free labels (`difficulty`, `category`, `expected_topic`):

```json
{"query": "What changed about PEP 594 dead batteries?", "acceptable_documents": ["3.13.html"],
 "acceptable_headings": ["PEP 594: Remove"], "difficulty": "Easy", "category": "technical_docs"}
```

Relevance is judged by document and heading, never by chunk id, so a dataset survives re-chunking and re-embedding, and a query can have several valid sources. An experiment copies its queries into its own row and records the file's SHA-256, so editing the file later cannot change what an experiment measured. Empty or duplicate queries are rejected before anything runs.

### Relevance
A retrieved chunk is relevant when an acceptable document matches its source (a substring, or a contiguous run of the document's alphanumeric tokens: `3.13.html` matches `.../whatsnew/3.13.html`). With acceptable headings, the match is *exact* when a heading also appears in the chunk's section title or heading path, and *partial* when only the document matched. Both count as relevant; *exact* also sets an exact rank. These are the previous harness's rules unchanged, so results stay comparable.

### Experiments
A spec file describes one experiment; each trial is a full `RetrievalConfig` (Phase 5):

```json
{
  "name": "dense vs hybrid vs reranked",
  "dataset": "evaluation_datasets/my_queries.json",
  "tenant_id": "demo",
  "baseline": "dense",
  "trials": {
    "dense": {"strategy": "dense"},
    "hybrid": {"strategy": "hybrid"},
    "hybrid+flashrank": {"strategy": "hybrid", "reranker": "flashrank", "rerank_candidates": 20}
  },
  "concurrency": 4,
  "generation": {"judge": {"provider": "groq", "model_name": "openai/gpt-oss-120b"}}
}
```

```
python -m src.evaluation run spec.json [--json report.json] [--fail-on-regression]
python -m src.evaluation resume <experiment_id>
python -m src.evaluation report <experiment_id> [--details] [--json report.json]
python -m src.evaluation list [--tenant demo]
```

- **One workspace.** An experiment searches one tenant's chunks, like any other search. The old harness searched all tenants at once through an "allow global" switch; that bypass is gone, so tenant isolation now has no exception.
- **Stored per query.** Each run row holds the rank of the first relevant chunk, the exact rank, the rank before reranking, the retrieved chunks with their match labels, latency, the costs, and why the run was degraded or failed. Reports are computed from these rows, so they can be rebuilt or drilled into at any time.
- **Resumable.** A query with a valid run is never re-run. A missing, degraded or failed run is, and the retry overwrites every column of the failed attempt. Re-running a finished experiment does nothing.
- **Bounded concurrency.** Up to `concurrency` queries of a trial run at once in one event loop (the previous harness ran one at a time). Latency percentiles are therefore measured under that concurrency; set it to 1 for latency-faithful numbers.
- **Fair costs.** Trials share retrievers, so each query is embedded once per index across all trials. Each run nevertheless records what *its* embedding costs, cached or not; otherwise every trial after the first would look cheaper for reusing the cache.
- **Conditions recorded.** The experiment records the concurrency, the generation and judge models, the cache counters, and whether an index changed size between a pause and a resume. Each trial records its index and the index size when it started.

### Metrics
- **hit_rate@k**: the share of queries with a relevant chunk in the top k (success@k), for k in 1, 3, 5, 10 up to the trial's `top_k`. Earlier reports called this "Recall@k"; the two coincide only when a query has one relevant source.
- **mrr**: the mean of 1/rank of the first relevant chunk (0 when none).
- **faithfulness** (generation only): the judge's 0 / 0.5 / 1 score of the answer against its context.
- Latency (mean, p50, p95 by nearest rank), and cost per query (embedding + rerank + generation).
- Breakdowns by `difficulty` and `category`.
- **Reranker forensics**: for reranked trials, how often reranking promoted, demoted, kept or lost the first relevant chunk relative to the first-stage pool it reordered.

Metrics use **valid runs only**: a degraded run (Phase 5: hybrid served sparse-only, or a reranker failed) or a run with an error did not run the configuration it names. Such runs are counted and shown, never mixed in.

Hit rate, MRR and the significance test are checked against [ranx](https://github.com/AmenRa/ranx) in the test suite. ranx itself is not a runtime dependency: it pulls in numba, pandas and seaborn, hundreds of megabytes for about 60 lines of arithmetic.

### Significance
Every trial is compared with the baseline on each metric, over the queries valid in both:

- **Paired randomization (sign-flip) test** on the per-query differences. If A and B were interchangeable, each difference would be as likely negative as positive. The p-value is the share of sign patterns with an absolute sum at least as large as the observed one. It assumes nothing about the metric's distribution, which matters because per-query hit rates are 0/1 and reciprocal ranks are lumpy. The test is exact (every pattern enumerated) when at most 13 queries differ, and otherwise uses 10,000 patterns from a fixed seed, so a report is reproducible.
- **Holm correction** within each metric's family of comparisons: with several trials against one baseline, the chance of any false "better" or "worse" stays at `alpha` (0.05 by default).
- **Verdicts**: `better`, `worse`, `no significant difference`, or **`insufficient evidence`**. Only the queries where the two trials differ carry information, and m differing queries can never produce a p-value below 2/2^m. With fewer than 6 differing queries, no difference can reach p < 0.05, so the report says the test set is too small to tell instead of claiming there is no difference.

`--fail-on-regression` exits 1 when any trial is significantly worse than the baseline. The CI `retrieval-gate` job (manual) runs a spec this way.

### Generation and Judging (optional stage)
With `generation` in the spec, each valid run is also answered (Phase 7's context building and prompt) and judged for faithfulness:

- **Pinned models, no fallback.** A fallback model mid-experiment would change what is being measured. A failed generation makes that run invalid, and it is retried on resume. A judge that cannot be called (rate limit, outage) **pauses** the experiment instead of letting a different judge score the rest; resume it once the provider recovers. A judge reply that is not a usable score invalidates only that run.
- **Caches in Postgres.** An answer is keyed by (tenant, model, full prompt) and a verdict by (tenant, metric, judge model, question, answer, sorted context chunk ids). Configurations that retrieve the same context build the same prompt, so it is generated and judged once: for example, two `top_k` values that end with the same chunks after reranking. Hit counts are recorded with the experiment. A cached answer still carries its original cost, so each configuration shows what it costs. The tenant is part of every key, so cached content never crosses workspaces.
- **Judge choice.** Defaults to the configured chat model. A judge from a different model family than the generator is better practice (self-preference bias). The judge is the single-score faithfulness prompt for now; RAGAS metrics are item 12.

## Prototype-Corpus Results (retired 2026-09-28)
The first corpus (2,284 chunks across about 60 documents) was deleted as prototype data, along with its frozen baselines and the file-based harness that measured them. Findings worth keeping:

- **Keyword-only search was fast but weak on paraphrase.** Sparse-only scored Recall@5 0.342 at ~7 ms, against dense-only 1.000 at ~1.8 s (the dense time was dominated by the hosted query embedding). Hybrid matched dense on recall.
- **Reranking trades top-1 precision for top-5 coverage.** On 38 queries, hybrid with the Jina reranker gave Recall@1 0.816 against 0.974 without it, but Recall@5 1.000 against 0.974. (Runs on a 6-query sample were too small to support conclusions.) FlashRank has not yet been measured.
- **A benchmark near its ceiling cannot see real defects.** Hybrid fusion once never merged a chunk found by both retrievers (Phase 5), and the keyword index covered about half the corpus. Fixing both left Recall unchanged at 0.9737, yet 10 of 38 queries got a different top-1 chunk and 14% of top-5 slots had held duplicates. Recall was at 0.97–1.00 and scored by document, not chunk. Harder queries and chunk-level relevance are needed, which is the purpose of synthetic test sets (item 11).
- **The Postgres migration preserved retrieval exactly.** Copied `halfvec` vectors reproduced the dense baseline to four decimals (worst-case stored vs original cosine 0.99999998).
- **Synthetic-query bias.** That benchmark was LLM-generated with the corpus's own vocabulary, which likely inflated absolute scores. Treat any such benchmark as a tool for relative comparison.

## Design Philosophy & Tradeoffs
- **Evidence over point estimates.** A 2-point difference on 40 queries is often noise. The report always gives the paired sample size, the number of queries that actually differ, and an adjusted p-value next to the means.
- **Judge fallibility.** An LLM judge can be wrong; pinning it per experiment at least keeps it consistently wrong, so comparisons between trials stay fair. The single 0 / 0.5 / 1 faithfulness score is coarse.
- **Document-level relevance.** Judging by document and heading tolerates re-chunking, but it cannot tell two chunks of the right section apart. Chunk-level ground truth arrives with synthetic test sets (item 11).
- **Concurrency vs latency fidelity.** Concurrent queries finish an experiment faster but contend for the same CPU, database and rate limits, which inflates latency. The concurrency is recorded with every experiment.
