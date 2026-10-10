# Competition-shaped local benchmark

This harness evaluates the participant-controlled part of AML: synchronous
`Add` followed by ranked evidence-only `Search`. It deliberately does not call
an Answer model, so ordinary retrieval regression runs have no model cost.

## Quick regression

Start a development endpoint with provenance enabled, then run:

```bash
python3 scripts/run_http_benchmark.py \
  --dataset synthetic \
  --synthetic-scale small \
  --base-url http://127.0.0.1:8080 \
  --max-probes 200
```

For an authenticated endpoint, export `AML_API_KEY` in the shell. The key is
read from the environment and is never written to the report.

## Public LoCoMo data

```bash
python3 scripts/fetch_benchmark_data.py locomo
python3 scripts/run_http_benchmark.py \
  --dataset locomo \
  --data-path benchmark_data/locomo10.json \
  --base-url http://127.0.0.1:8080 \
  --max-probes 200
```

The data directory and generated reports are git-ignored. LoCoMo remains under
its upstream terms and attribution; this repository does not redistribute it.

## LongMemEval-S

The cleaned S release is roughly 265 MB. The adapter reads its top-level JSON
array incrementally, but every selected question still contains a large private
haystack and is indexed under an isolated user ID. Start with 20 questions:

```bash
python3 scripts/fetch_benchmark_data.py longmemeval-s
python3 scripts/run_http_benchmark.py \
  --dataset longmemeval-s \
  --data-path benchmark_data/longmemeval_s_cleaned.json \
  --base-url http://127.0.0.1:8080 \
  --max-probes 20
```

The adapter uses turn-level `has_answer` annotations when present. Compatible
exports containing only `answer_session_ids` fall back to one stable message
from the annotated session and record that fact in case metadata; those fallback
scores must not be mixed silently with turn-annotated results.

## Contract alignment and limits

- Each run uses fresh `user_id` values to avoid cross-run contamination.
- Each source session is split at 20 messages or an approximate 2,000-word
  boundary, then persisted before Search.
- `top_k` defaults to the formal AML value of 100.
- Evidence is matched by normalized source text, not private database IDs.
- Recall@5/20/100, MRR, forbidden-evidence leakage and HTTP latency are reported.
- This is a retrieval proxy. It does not reproduce the platform's hidden data,
  locked Answer model, judge, prompts, or final aggregate score.

The public AML adapter's exact word counter is not distributed. This harness
counts Latin words/numbers and individual CJK characters deterministically and
labels that behavior as an approximation.
