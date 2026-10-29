# Source-aware result ordering (experimental)

Disabled by default: set flags.source_aware_dedup=true explicitly.

After the existing pair/semantic ranking, stably defer a candidate only when an
EARLIER RETAINED SINGLE candidate has both (1) a superset of its original source
message IDs and (2) its complete whitespace-normalized text as a substring.
Distinct source IDs, partial text overlap, new context and unknown provenance are
retained. A union of separate messages cannot replace a joint-context document.
No role or recency filtering, semantic threshold, model call, deletion or score
change is used. All original candidates remain: retained prefix then deferred
tail, each in original relative order. Scores remain upstream relevance scores;
array order is the final rank. Pre-rerank dedup and source slot guarantees stay
unchanged. The opt-in final ordering may change view proportions in short prefixes.

Provenance records original rank and the retained covering witness; flags mark
deferred candidates. Any selector failure keeps original order and marks fallback.
The function is request-local, has no cache/DB reads, and requires one tenant pool.

Evaluation must separate ranking metrics, complete-marker coverage and duplicate
occupancy. The strict condition reduces an opportunity for information loss but
does not prove answer quality, semantic equivalence or production latency.
Current offline experiment reuses frozen public candidate pools and upstream model
scores; it is not fresh inference, a new holdout or full Search/Answer evaluation.
