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
  --longmemeval-per-capability 4 \
  --max-probes 24
```

This selects four questions from each of LongMemEval's six capabilities instead
of taking the first rows, which are grouped by question type in the public file.

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

## Additional public benchmark adapters

The same HTTP-shaped retrieval runner also accepts the following datasets:

| Dataset | `--dataset` | Required local inputs | Retrieval score |
| --- | --- | --- | --- |
| BEAM | `beam` | Official JSON/JSONL export | Latency only; final rubric score needs an Answer model |
| CL-bench | `clbench` | Official JSON/JSONL records | Answer model required |
| HaluMem | `halumem` | `HaluMem-Medium.jsonl` or `HaluMem-Long.jsonl` | Evidence proxy when question evidence is present |
| LoCoMo Refined | `locomo-refined` | Annotated JSON, or conversations plus `--questions-path` | Evidence proxy when public evidence is present |
| PersonaMem-v2 | `personamem-v2` | Benchmark CSV plus `--chat-history-dir` | Answer model required |
| ScriptMem | `scriptmem` | A legally obtained local bundle containing the source dialogue | Answer model required |

Example adapter and HTTP run:

```bash
python3 scripts/fetch_benchmark_data.py halumem-medium
python3 scripts/run_http_benchmark.py \
  --dataset halumem \
  --data-path benchmark_data/HaluMem-Medium.jsonl \
  --base-url http://127.0.0.1:8080 \
  --max-probes 20
```

LoCoMo Refined's two public files can be fetched together:

```bash
python3 scripts/fetch_benchmark_data.py locomo-refined
python3 scripts/run_http_benchmark.py \
  --dataset locomo-refined \
  --data-path benchmark_data/locomo_refined_conversations.jsonl \
  --questions-path benchmark_data/locomo_refined_questions.jsonl \
  --base-url http://127.0.0.1:8080 \
  --max-probes 20
```

BEAM's Hugging Face release is Parquet. Convert a selected split to JSON or
JSONL with the upstream `datasets` library before passing it to this standard-
library-only runner. The adapter supports the official 2-D chat batches,
`turns` batch dictionaries, flat turn lists, and the 10M plan layout. It also
parses the official Python-repr `probing_questions` field without executing it.

ScriptMem intentionally excludes the original source dialogue for copyright
reasons. The adapter refuses the question-only public placeholder instead of
reporting a misleading successful retrieval run. Do not commit benchmark data,
licensed source text, model credentials, or generated reports; their directories
are ignored by Git.

These adapters exercise the participant-controlled Add/Search layer. They do
not claim parity with each upstream benchmark's Answer generation, judge model,
or official final metric.
