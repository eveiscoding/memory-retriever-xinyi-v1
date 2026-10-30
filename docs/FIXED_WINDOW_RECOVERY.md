# Fixed-window HTTP recovery profile

The explicit fixed-window service uses profile fixed-parent-384-64-max-source-8k-query256-v2.
The default service and offline ParentWindowRetriever are unchanged.

Lexical and rule recall receive the complete original query and options. Only model
inputs are bounded to 256 tokens using original head/tail text. The complete question
has priority over options; options fill remaining room only. Parent scoring retains
its query-plus-options semantics; source scoring retains question-only semantics.
Both use the same budget helper. Role prefixes for source scoring are bounded to
16 tokens; original role metadata and source text remain unchanged. Short inputs
that fit the budget preserve their exact model text, scoring, RRF and 8K rendering.
Long input answer quality has not been established by this reliability repair.

Parent-window scoring failure falls back to legacy bounded parent scoring, then
existing deterministic recall ordering if that also fails. Source window/model
failure returns a budgeted prefix of validated original messages. Such responses
include top-level degraded=true. No aggregate text is substituted as original text.
Source/provenance/tenant validation or packing integrity failures return 503 and
never successful unvalidated evidence. Validation errors remain 422.

Logs contain a random request correlation ID, stage, exception class, numeric counts
and actual strategy. They never include exception messages, query/options, user or
source IDs or evidence. Request and model locks cover the entire Add/Search snapshot;
request context is reset after each call. No new API/model/embedding is introduced.

Verification: unittest discovery (including recovery fixtures), real frozen ONNX
boundary HTTP probes, short-query byte-for-byte frozen-service comparison, and
16 concurrent Search plus same-user Add on disposable SQLite and loopback HTTP.
These are reliability checks, not an offline quality evaluation or official Smoke.
