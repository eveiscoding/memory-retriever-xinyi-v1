# Parent-internal window scoring experiment

Explicit Python subclass ParentWindowRetriever; default RetrieverDB, Add, HTTP
and indexes remain unchanged. Preserve the existing pre-pair candidate IDs,
content and order. Original message inputs use the existing scorer and truncation.
Each aggregate parent is split into exact contiguous original parent-text spans,
up to384 reference tokens with64 overlap, query-adapted under512 pair tokens.
All spans are scored; max logit represents the parent, ties take earliest span.
Same original RRF .25/k60 combines ranks, then same source-aware visible guard.
Only the winning bounded span is marked visible for each parent; never the full
parent. Short blocks are not indexed or returned as standalone candidates.

The winning span can cross message boundaries and start inside a message. This
first experiment holds chunking simple to isolate candidate preservation. Max
scoring can favor long parents with many windows. Record model-pair counts and
latency; this is not an equal-cost or calibrated-probability comparison.

Any invalid tokenization, scoring, visibility or window-budget result atomically
falls back to legacy parent scoring with parent_window_fallback. The experiment
requires zero fallbacks. Runs use isolated frozen public indexes, not production.
The per-instance parent_window_trace is offline diagnostic state, not a public
concurrent request interface. No production deployment or lifecycle change.
