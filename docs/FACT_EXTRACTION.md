# Additive fact extraction (logic-only, experimental)

This implements the **pipeline**, not a trained fact extractor. No model, model
SDK, paid API, network request, or automatic model download is included. It is
disabled by default. A configured backend alone does not enable it.

Inspired by Mem0's [focused extraction instructions and structured facts output](https://docs.mem0.ai/open-source/features/custom-instructions)
and [add/infer distinction](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/memory-operations/add.mdx),
accessed 2026-10-21. This is an independent implementation, not a Mem0 dependency,
port, or assertion of equivalent behavior. We adopt atomic additive facts and
irrelevant-input filtering; we do not implement Mem0's complete memory system.

## Boundary

`Add → normalize messages → existing-request check → backend → validate all facts
→ commit raw messages, views, facts, citations, and extraction outcome together`.

- Original messages and views remain unchanged and searchable, even when extraction
  fails, returns no facts, or exceeds the input budget. Database write errors still
  roll back the entire Add rather than acknowledging a partial write.
- Facts live in separate `facts`, `fact_sources`, and `fact_extractions` tables.
  They are **not** indexed in FTS or vectors, fed to RRF, or exposed by Search.
  Thus this stage supports storage/audit, not improved fact retrieval yet.
- `db.list_facts(user_id=..., limit=100)` is an internal scoped audit method.
  It returns derived text plus original message IDs, exact quotes, roles, source
  timestamps, session IDs, and request IDs. There is no new HTTP route.
- Deduplication is user+speaker+NFC/whitespace-normalized text only, case-sensitive.
  Every duplicate can add source links without duplicating a fact. This is NOT
  semantic deduplication, contradiction resolution, UPDATE, or DELETE.
  Different/negated facts coexist; no current-truth or temporal-validity claim.
- User deletion, full purge, and TTL cleanup remove the derivative tables too.

## Backend contract

```python
class MyExtractor:
    identity = "provider-model-prompt-revision"  # non-secret, <=128 characters

    def extract(self, *, messages, instructions, max_facts):
        # Future provider integration must enforce a timeout and response-size
        # limit, use approved credentials, and avoid logging inputs/secrets.
        return {"facts": []}
```

The empty example demonstrates the interface only; it does not extract facts.
`messages` contains copies of only user/assistant turns from **this Add** (not
other users, prior sessions, or a query). Each includes original `message_index`,
`role`, `content`, and `timestamp`. System/developer/unknown roles are excluded.
`instructions` is versioned as `aml-facts-v1`; it asks for reusable facts without
invented attributes, retaining subjects, uncertainty, negation, numbers, and time
expressions. Named third parties must not be relabeled as the user. Instructions
inside source text are untrusted data. These are model instructions, not semantic
security guarantees. Missing cross-chunk context is a known limitation.

Return a JSON object (or plain JSON string, no Markdown fences):

```json
{"facts":[{"text":"The user prefers tea.","kind":"preference",
"sources":[{"message_index":0,"quote":"I prefer tea."}]}]}
```

Allowed kinds: `profile`, `preference`, `event`, `plan`, `constraint`, `knowledge`.
All citations in one fact must come from a single role; mixed roles are rejected.
Attribution is derived from sources, never accepted as a model-supplied user ID.
The model cannot supply database IDs, tenant IDs, timestamps, update/delete
operations, or metadata. Every source index must reference this Add and every
quote must occur exactly in that original message. The fact text itself may be
a paraphrase: citation validity **does not prove entailment, correctness, or that
all relevant facts were extracted**. Those require later real-model evaluation.

The entire output batch is rejected if any item fails validation. Limits:
1,024 characters/fact, 1–8 citations/fact, 2,048 characters/quote, 65,536
characters for JSON-string output, plus configured fact/input budgets. Backends
returning objects must bound their transport/output before constructing them.

## Enable only in isolated tests

Inject `fact_backend=MyExtractor()` into `RetrieverDB`, or configure a trusted local
`module:factory` with `AML_FACT_BACKEND`. Also explicitly set
`AML_FLAG_FACT_EXTRACTION=1`. Enabled-without-backend fails startup; disabled never
loads or calls the backend. No built-in rule/model backend is supplied.

`AML_FACT_MAX_INPUT_CHARS` defaults to 16,000 (1–1,000,000),
`AML_FACT_MAX_FACTS` defaults to 32 (1–256). Input budget counts eligible source
content and skips oversized batches rather than silently truncating evidence.
Fact-enabled Adds serialize extraction to prevent repeated calls on simultaneous
idempotent retries. Provider execution is outside the SQLite write transaction;
a future slow provider still increases Add latency and needs capacity testing.

Internal `AddResult` adds `fact_status` and `fact_ids`; public Add echo fields and
success semantics are unchanged. Status is `disabled`, `extracted`, `no_facts`,
`input_limit`, or `failed`. Failed output/transport persists only generic
`invalid_output`/`backend_error` codes, never exception text. Extraction records
include backend identity and prompt version. Raw-only requests have no fact row.

Retries return the original outcome without extraction; reusing a request key
with a different payload still conflicts. Enabling extraction does not silently
backfill old Adds, and retrying a previously failed extraction does not retry a
provider. Explicit backfill/retry controls remain future work.

## Verification and next steps

Run targeted logic tests with `python -m unittest tests.test_facts -v`.
Fixtures are simulated structured output, not model-quality evidence.
Next: select and authorize a generation backend; assess extraction precision,
coverage, citation entailment, speaker/subject errors and cost on locked public
data; only then consider a separately gated fact recall channel. Time/version
management and complementary recall are subsequent independent stages.
