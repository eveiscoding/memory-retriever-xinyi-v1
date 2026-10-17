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
2. Freeze membership after dedup and slots. Keep baseline order through the
   configured boundary (default 20); only ranks 21–100 are semantically reranked.
   Earlier front-bucket variants failed public quality gates. This conservative
   tail-only mode does not claim to improve Top20 ranking or reasoning.
3. With boundary=20 and top_k=100, all Top5/20 IDs, ranks and source coverage are
   unchanged. Top100 membership also stays fixed. Tail segments retain up to
   three exact matches with disjoint sources and distinct normalized content.
   Entity escape eligibility filters How/Which and other query stop words;
   baseline entity scoring is unchanged. No candidate is inserted or discarded.
   There is no benchmark-label, gold-evidence or query-intent routing. Smaller
   configured boundaries weaken the frozen-prefix guarantee; an extra cutoff at
   5 still preserves Top5 membership inside the pool. Separate top_k calls retain
   independent baseline message slots, without a prefix-consistency promise.
4. Failures return baseline order with `semantic_rerank_fallback`. Indexing/Add
   failures still follow the existing transaction behavior; they are not masked.
5. `AML_INCLUDE_PROVENANCE=1` exposes pre/post ranks and `semantic_score`.
   Existing `score` remains the baseline RRF score, so it need not be monotonic
   after semantic reordering. Consumers must respect the returned list order.

Disable `AML_FLAG_SEMANTIC_RERANK` to roll back. Keep vector=0 to restore the
no-dense path. Protection eligibility alone filters query stop words; entity features in baseline
scoring are unchanged. Membership guarantees use the same input and IDs.
The semantic feature remains disabled by default. The strict paired gate requires no per-gold coverage loss at 5/20/100 or
aggregate/capability regression. Tail-only gains can be small and do not establish
the roadmap target of improved preference/multi-session Top20 ranking. That
requires further work and independent validation; the feature stays experimental. Unit fixtures prove the
invariants only, not real-model quality.
