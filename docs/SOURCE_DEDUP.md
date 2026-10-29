# Source-aware ordering with visible-input protection (experimental)

Disabled by default: flags.source_aware_dedup=true opts in. It requires reliable
pair-backend visible_spans(query, documents) support; unknown/invalid visibility
keeps original pair order and marks source_visibility_unavailable. Pair inference
success is retained if the optional span provider fails.

One earlier retained SINGLE witness must contain both the original source ID set
and the complete target body (whitespace-normalized, case-sensitive), and its
actual query/model input span must contain that complete body as well. New facts,
new source IDs, joint context and compact facts beyond a long view's visible
prefix are retained. No role/time/embedding threshold is used.

ONNX visible_spans repeats exactly the score tokenizer options, excluding query
and padding offsets. It adds tokenizer work only, no graph inference. The pooled
backend leases a tokenizer/session owner for this operation; no mutable last-call
trace or cross-query cache is used. The retriever validates aligned bounds under
its pair lock and attaches request-local metadata by original document ID.

All original documents, content and scores remain; only final array order may
change. Scores remain upstream relevance scores. Short-prefix source proportions
and full-text budget may change; evaluate them separately. Output provenance
records covering witness and its visible interval, not raw text or credentials.

Frozen-pool replay is not fresh inference, a new holdout or full Search/Answer
validation. Both complete-original gold ranking and actual-input marker coverage
must be checked. Marker coverage does not establish semantic or answer accuracy.
