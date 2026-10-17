# Phase 03: protected semantic rerank (experimental)

Priority: P1 for offline validation, not a production promotion. This implements
the bounded rerank stage of the open-source memory pattern, not fact extraction,
graph retrieval or a dedicated cross-encoder. Scores are embedding cosine.

## Enable explicitly

Use the existing Singapore credentials and embedding backend:

```sh
export AML_DENSE_BACKEND=aml_retriever.dashscope_embedding:create_backend
export AML_DENSE_DIMENSION=1024
export AML_FLAG_VECTOR=0
export AML_FLAG_SEMANTIC_RERANK=1
export AML_SEMANTIC_RERANK_POOL=100
export AML_SEMANTIC_RERANK_BOUNDARY=20
export AML_SEMANTIC_RERANK_EXACT_QUOTA=3
```

Do not enable against an unindexed large database casually: initialization still
backfills document embeddings synchronously and Add still embeds documents. With
an existing matching index, Search embeds one query, reuses persisted vectors,
and caches 128 query/pool score tuples in memory. Keys hash the user, model, query
and candidate IDs/content. Query/content text is not retained in the score cache.
Provider calls can incur fees. No GPT generation or new model is introduced.

## Exact guarantees and limitations

1. Freeze the no-dense FTS + feature baseline before scoring, dedup and message
   slots. Even if vector=1, semantic mode bypasses dense candidate generation.
2. Freeze the final result membership after dedup and slots, then sort separately
   inside ranks 1–20 and 21–100. The first three exact numeric/date/entity/phrase
   matches with disjoint source IDs and distinct normalized content in each
   segment retain their positions. Repeated views do not consume extra escape
   slots. Different messages are not assumed to be the same event. No record is
   discarded or injected. Entries beyond the pool stay put.
3. For an identical top_k=100 baseline call, candidate IDs and source coverage at
   20 and 100 are identical by construction. Other cutoffs (notably 5) can change.
   For interval/order queries, an additional boundary at 5 preserves baseline
   Top5 membership (segments 1–5, 6–20, 21–100 with default settings). This protects
   multiple historical events without guessing gold evidence or boosting recency.
   Query intent is inferred from the query alone, never options or dataset labels.
   Ordering within each segment can still change; MRR/nDCG may still regress.
   Different requested top_k values have independent baseline slot guarantees;
   this is not a promise of prefix consistency across separate Search calls.
4. Failures return baseline order with `semantic_rerank_fallback`. Indexing/Add
   failures still follow the existing transaction behavior; they are not masked.
5. `AML_INCLUDE_PROVENANCE=1` exposes pre/post ranks and `semantic_score`.
   Existing `score` remains the baseline RRF score, so it need not be monotonic
   after semantic reordering. Consumers must respect the returned list order.

Disable `AML_FLAG_SEMANTIC_RERANK` to roll back. Keep vector=0 to restore the
no-dense path. Entity extraction now filters query stop words, including How/Which;
this also affects deterministic retrieval when semantic rerank is off. Membership
guarantees compare on/off calls using the same code, not different code versions.
The semantic feature remains disabled by default. The gate is actual preference
and multi-session MRR/nDCG improvement, preserved Recall@20/100, no update-class
regression and acceptable controlled Search latency. Unit fixtures prove the
invariants only, not real-model quality.
