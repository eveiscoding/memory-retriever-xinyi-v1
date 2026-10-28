# Experimental numeric fact support

This opt-in Search-only experiment reranks the already returned pool after the
existing semantic/pair stage. It never changes Add, the schema, candidate IDs,
options, raw evidence, MiniLM inputs, clocks, temporal rules or production flags.
It is NOT an answerability classifier, entailment model or quality promotion.

## Arms

* B0: AML_FLAG_FACT_SUPPORT=0 (default).
* Rules: AML_FLAG_FACT_SUPPORT=1, AML_FACT_SUPPORT_MODE=rules.
* Embedding: same flag, AML_FACT_SUPPORT_MODE=embedding, plus an explicitly
  configured sentence embedding backend. No dense index or vector flag needed.

Example opt-in environment for an isolated public-data process:

    AML_FLAG_FACT_SUPPORT=1
    AML_FACT_SUPPORT_MODE=embedding
    AML_FACT_SUPPORT_BACKEND=aml_retriever.dashscope_embedding:create_backend
    AML_FACT_SUPPORT_DIMENSION=1024
    AML_FACT_SUPPORT_WEIGHT=0.1
    AML_FACT_SUPPORT_K=60
    AML_FACT_SUPPORT_MAX_CHARS=32768
    AML_FACT_SUPPORT_PER_DOC=4
    AML_FACT_SUPPORT_MAX_TOTAL=400
    AML_FACT_SUPPORT_MIN_SIMILARITY=0.25

The existing text-embedding-v4 adapter separately requires DASHSCOPE_API_KEY and
its workspace-specific DASHSCOPE_BASE_URL. Supply secrets through the established
private environment, never CLI arguments or tracked files. This feature does not
reuse a Full service's environment or authorize sending official data to an API.
Hashing and injected test backends only verify mechanics, not semantic quality.
No live provider call or public-data quality result is implied by the unit tests.

## Scoring

Only recognized English quantity, money and duration questions are considered.
Conjunctions/comparisons yield up to eight query-only needs. Unknown or oversized
queries leave the old result exactly unchanged. There are no answer values,
case IDs, dataset imports, geographic alias tables or gold labels in the policy.

A bounded candidate prefix is split into short clauses. Eligibility requires a
personal factual assertion, an attribute-compatible value, matching explicit
proper-name anchors and any recognized money domain (lodging, meals, transport).
Questions, negation, modal suggestions and plans are conservatively skipped.
Simple third-person kinship subjects cannot stand in for a self question.
Assistant restatements are allowed; role alone neither rewards nor penalizes.
This is an imperfect English heuristic, not a general relation extractor.

Rules score is query-anchor coverage within an eligible clause. Embedding score is
max(0, (cosine - minimum_similarity) / (1 - minimum_similarity)), still subject to
the same assertion/attribute/name gates. Embedding can bridge common-noun
paraphrases; it cannot bypass an explicit country/name mismatch. A geographical
alias such as a city versus a country is intentionally unresolved in v1.
Both arms use the same lexical clause preselection and work budgets. Per-document
support is the maximum over its chosen clause/need pairs, not a requirement for
one document to contain all facts. This does not guarantee joint evidence coverage.

Fusion uses the rank of B0's final returned order, NOT its legacy score field:

    final_i = (1-beta)/(k + B0_rank_i)
              + beta*support_i/(k + support_rank_i)

Equal support has equal support rank; zero support has zero bonus. Ties retain B0
order. No or constant support and beta=0 are exact no-ops. The underlying score
field stays diagnostic, as in pair reranking; consumers must respect list order.
These defaults are implementation starting points, not tuned acceptance values.
Do not vary beta against the eleven diagnostic cases and call it held-out evidence.

## Bounds, privacy and failure behavior

Each document scans at most 32768 characters, skipping an incomplete final clause
and clauses over 800 characters. At most four eligible clause/need pairs per
document, 400 across the request, are selected round-robin. Unique texts are
embedded once per request with up to eight need texts. No cross-user/request cache
or persisted embeddings are created by this feature. The optional backend runs
under its own serialized lock, outside SQLite locks. The existing provider's
batch size, timeouts and retries apply; live API latency/cost still need measuring.

Provider failure, malformed vectors, zero vectors or non-finite values atomically
retain B0 order, with a fact_support_fallback flag and no error-body logging.
No clause text is added to traces. Internal provenance includes support, clause
index, mode, selected-pair count, scan-truncated flag and backend identity/hash.
The official API evidence shape is unchanged. Existing evidence content remains
in the normal authorized response.

Full-document clause scanning is a NEW bounded signal shared by Rules/Embedding;
it may see text beyond MiniLM's 512-token prefix. Do not interpret an improvement
as fixing MiniLM truncation or changing its input. Time updates, refusal decisions,
Chinese coverage and full Answer/Packing behavior remain outside v1.

## Validation and proposed effect comparison

    python3 -m unittest tests.test_fact_support -v
    python3 -m unittest discover -s tests -v

Tests use synthetic facts, negative cases, malformed vectors, tenant isolation,
no-Add-call assertions, pair combination and mock text-embedding-v4 HTTP responses.
They do not use the diagnostic golds to implement or tune the rules.

For a later separately locked quality run, compare B0 / Rules / Embedding on the
same inputs, returned pool, clocks, pair logits/model and sentence budgets. Record
real backend/version, dimensions, calls, latency, fallback rate, Recall5/20,
Complete20, gold-MRR and every newly lost gold. Audit abstention/weak annotations,
unannotated equivalent evidence and old/current values separately without changing
historical golds. Do not claim answer accuracy, throughput improvement or promotion
from mock tests or replayed rankings. No such quality run is launched by this change.

## Delivery check (2026-10-28)

A separate synthetic live check reused the existing private embedding validation
environment without reading or changing Full configuration. One text-embedding-v4
request encoded two texts (a synthetic question and eligible factual clause),
returned 1024-dimensional vectors and exercised the soft order. This is only a
connectivity/functional check, not public-case quality evidence. The original
report is kept outside Git at /home/ubuntu/aml-fact-support-live-20261028.json.
