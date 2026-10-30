# Fixed-window 8K explicit service

Run python -m aml_retriever.fixed_window_service with the frozen MiniLM model,
context512/batch1/threads1, pair RRF .25/k60/pool100 and source-aware dedup.
Default HTTP service and CLI are unchanged. No embedding or generation.

Parent scoring is exactly branch22: original messages once, aggregated parents
384-token windows/64 overlap/query-adaptive, maximum logit, earliest tie.
Top20 sources are resolved and independently scored by the existing frozen 8K
consumer. Official Search returns one item containing its exact complete rendered
8192-reference-token evidence pack, including IDs/roles/times/spans. top_k=0
returns empty; smaller positive top_k limits source parents. No extra unbudgeted
content is appended. No source is read from another user or session.

A service-level lock serializes Add/Search for tokenizer ownership, request-local
traces and stable source resolution. This conservative first service uses one
ONNX session and no concurrent model calls. Retention cleanup is unchanged.
Search fallback is rejected rather than silently evaluating another strategy.
Fail startup on mismatched model/profile. Smoke is not an offline promotion.

Use an independent empty volume. Keep previous Full deployments immutable.
Bind this version only after the user selects their intended leaderboard key.
