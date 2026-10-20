# Experimental cross-encoder relevance reranking

This Phase03 ablation scores each (query, evidence content) pair jointly instead
of comparing independently embedded vectors. It does not call a generative LLM,
classify benchmark categories, use gold labels, or recognize special query words.
It does not extract events or guarantee multi-event completeness.

## First experiment

Use `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, English MS MARCO passage
ranking checkpoint), provisioned on the isolated server at a fixed upstream
revision. This small CPU model is a feasibility baseline, not a multilingual or
memory-domain quality claim. Source/model card:
https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

Optional isolated dependencies: CPU `torch==2.5.1`, `transformers==4.48.3`.
Provision safetensors/tokenizer/config artifacts explicitly before starting:

```sh
export AML_FLAG_VECTOR=0
export AML_FLAG_SEMANTIC_RERANK=0
export AML_FLAG_PAIR_RERANK=1
export AML_PAIR_BACKEND=aml_retriever.local_pair_backend:create_backend
export AML_PAIR_MODEL_PATH=/absolute/path/to/locked/model-snapshot
export AML_PAIR_RERANK_POOL=100
export AML_PAIR_MAX_LENGTH=512
export AML_PAIR_BATCH_SIZE=8
export AML_PAIR_THREADS=1
```

Default remains off, with no new mandatory dependencies. Runtime loads only local
files, requires safetensors, disallows remote code and downloads. Actual artifact
SHA256s and token limit identify the model configuration; record upstream revision,
dependency versions, hashes, device, and threads with each evaluation.

The existing no-dense candidate pipeline, baseline scoring, deduplication, and
message slots run first. Up to 100 returned records are sorted by relevance logit;
no records are added/deleted. This deliberately permits Top5/20 membership changes:
coverage is an empirical gate, not guaranteed by freezing the prefix. Baseline
`score` stays unchanged; diagnostic `semantic_score` is a raw cross-encoder logit
(not cosine/probability); pre/post ranks and `pair_rerank` flags distinguish modes.
Options participate using the existing recall text. Cross-encoder and old semantic
rerank are mutually exclusive. Pair mode bypasses dense recall, even if vector=1.

Inference is serial per RetrieverDB to bound CPU model memory. It does not hold
SQLite connections during inference or cache query/results. Failures/nonfinite or
incomplete scores retain baseline order with `pair_rerank_fallback`; startup model
failure stops startup rather than pretending the model loaded. Document suffixes
beyond the 512-token pair context are truncated; question text is preserved, and
an oversized question falls back. This can lose evidence inside long views.

## Validation and next hypotheses

Compare against the same code with both experimental flags off, independent
processes and fresh temporary DBs, identical user namespaces and input order.
Lock a public slice excluding all IDs from the previous 120-case development set
(and therefore the earlier smoke24). Freeze IDs/model/settings before reading
candidate results. That slice is holdout only for this decision, not evidence of
universal generalization or absence of pretraining contamination.

Report overall/capability Recall5/20/100, MRR, nDCG20, per-gold coverage/ranks,
fallback count and cold/warm latency. Strict existing quality gates apply; a
regression does not justify query-specific exceptions or tuning on the holdout.
Run the old smoke24 only as regression diagnosis. No official submissions or
production enablement are authorized by this experiment.

After this isolated relevance ablation, evaluate generic evidence-set selection
(relevance + redundancy + marginal coverage) as a separate hypothesis. Diversity
is not equivalent to answering every event. Time metadata representation follows
only if evidence shows it is needed; event time must not be replaced with record
time or invented when unknown. No Phase04/06 implementation is included here.

## Baseline + pair rank fusion (BGE experiment)

`AML_PAIR_RERANK_MODE=model` preserves the pure-model ablation. Set `rrf` for:

```
F(i) = (1-alpha)/(k+baseline_rank(i)) + alpha/(k+pair_rank(i))
```

`AML_PAIR_RRF_WEIGHT=0.25` and `AML_PAIR_RRF_K=60` are the predeclared initial
settings. No gold/query-intent routing or raw-logit calibration is involved.
All ranks start at one. Stable ties retain baseline order. Alpha=0 reproduces
baseline order, alpha=1 reproduces model order. The original pool stays fixed;
this does not guarantee Top5/20 coverage. Diagnostics retain raw `semantic_score`,
add `channel_ranks.pair_model`, and mark fused results with `pair_rrf`.

## Normalized score convex combination

Set `AML_PAIR_RERANK_MODE=cc` to compare rank fusion with a score-gap-preserving
alternative over the same fixed pool and pair logits:

```
F(i) = (1-alpha) * minmax(baseline_score(i))
     + alpha * minmax(pair_logit(i))
```

Normalization is per query and per scored candidate pool. `baseline_score` is
the existing pipeline's final baseline RRF score, not an original feature score;
the pair value remains an uncalibrated raw model logit. This mode does not treat
either input as a cross-query probability. `AML_PAIR_RRF_WEIGHT` is the shared
alpha knob for mutually exclusive `rrf` and `cc` experiments; `AML_PAIR_RRF_K`
is ignored by `cc` but still validated for configuration consistency.

A constant channel normalizes to all zero and contributes no discrimination. If
both channels are constant, stable ties preserve baseline order. Missing,
non-numeric, boolean, NaN, or infinite scores fail the whole rerank and trigger
the existing untouched-baseline fallback. Equal fused scores retain baseline
order. The returned membership remains fixed and no Top20/exact protection can
be combined with this mode. Diagnostics retain the raw `semantic_score`, model
rank, and mark successful results with `pair_cc`.

## Equal-weight three-channel convex combination

Set `AML_PAIR_RERANK_MODE=cc3` (example: `config.mminilm-cc3.env.example`):

```
F(i) = (minmax(bm25(i)) + minmax(rule(i)) + minmax(pair_logit(i))) / 3
```

This is a separate mode; historical `cc`, `rrf`, and `model` behavior is retained.
The three coefficients are fixed at 1/3, not selected from the RRF alpha setting.
Shared alpha/k are ignored here but still validated as finite valid configuration.
Baseline recall, candidate fusion, deduplication and message-slot selection run
unchanged first. Only the returned fixed pool (up to 100) is reordered; dense
recall remains excluded in pair mode, as before. This does not rebalance recall
or guarantee the baseline Top5/20 membership.

BM25 is the negated raw FTS5 rank (larger is better), with zero for candidates
without an FTS hit or from the LIKE fallback. The rule channel is recomputed on
copies of the same pre-fusion candidate records with an empty BM25 rank map.
This retains coverage, exact/entity/date/number, recency, conflict, preference,
and supersession rules, but excludes the BM25 numeric contribution. Crucially,
bounded supersession pair selection also uses the BM25-free rule scores; merely
subtracting BM25 after governance would not remove that dependence. Query options
and the baseline candidate population are unchanged. Rule overlap features may
still correlate with BM25; equal coefficients do not imply independent signals
or equal statistical influence.

Normalization is independently per query and scored pool, with constant channels
mapped to zero. A missing/boolean/nonfinite/malformed component rejects the entire
rerank and uses the existing baseline fallback. Stable ties retain baseline order;
an unscored tail is not changed. No Top20/exact protection can be combined with cc3.
The ordinary `score` retains the baseline diagnostic value and consumers must use
returned order. With provenance enabled, `fusion_scores` exposes raw
`bm25_raw/rule_raw/model_raw`, normalized `bm25/rule/model`, and `fused`;
`semantic_score` remains the raw model logit. Successful rows carry `pair_cc3`.
Provenance disabled omits all fusion diagnostics from the official response.

Equal weights are an experimental policy, not an established quality improvement.
Compare fixed public cases and model logits against baseline, RRF and the previous
two-channel CC. Previously inspected cases are development/regression material,
not new holdout. No official Smoke/Full or production change follows automatically
from enabling this local experiment.

The optional `aml_retriever.onnx_pair_backend:create_backend` loads an explicitly
provisioned local ONNX graph and tokenizer, using CPUExecutionProvider, with no
remote code or runtime downloads. `AML_PAIR_ONNX_FILE` defaults to
`onnx/model_int8.onnx`. Provision single-file weights (external tensor files are
not supported); lock actual artifact hashes, upstream revision, runtime version,
quantization, context length, batch size and thread count before evaluating.

Initial CPU feasibility candidate: `onnx-community/bge-reranker-v2-m3-ONNX`,
`onnx/model_int8.onnx`, derived from BAAI/bge-reranker-v2-m3. This is a community
INT8 conversion, not an original-precision BAAI result. Compare fusion and pure
order using the identical checkpoint, tokenizer and pair scores. Initial input
limit remains 512 tokens (only document suffix truncated), batch=1, threads=1;
this isolates fusion rather than also changing input coverage. The adapter can
accept explicitly chosen lengths through 8192, but extending context requires a
new registered experiment and actual model support. Do not extrapolate model-card
speed or multilingual scores to this machine or dialogue-memory task.

Sources:
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX
- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

Official competition FAQ 05 (checked 2026-10-19) explicitly says rerankers are
unrestricted; academic embedding is text-embedding-v4 and LLM components are
gpt-4o-mini. Search must still return evidence rather than generated answers:
https://agentmemories.ai/competition/

## Pure multilingual MiniLM deployment profile

`config.mminilm-pure.env.example` selects pure logit ordering, no RRF or dense
retrieval. Global defaults remain off. Build the optional CPU image using
`docker build -f Dockerfile.onnx -t aml-mminilm:local .`; mount the locked local
snapshot read-only at the profile's model path. Load the profile with `--env-file`
and supply authentication separately. Experimental Smoke requires a fresh DB.

Publisher: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), revision
`1427fd652930e4ba29e8149678df786c240d8825`. Use the publisher's INT8 AVX2 graph
`onnx/model_quint8_avx2.onnx`, SHA256
`6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b`.
Source: https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

Lock all tokenizer/config files as well. The tested snapshot's backend hash is
`781f3f03fe04bf180ba0d912b6ee7f0177cf793176b5a317a8a5e0e8e7036559`.
Verify it before routing. Pool=100/context=512/batch=1/threads=1 preserve the
public-data experiment's model settings. The 2-vCPU host took approximately
30 seconds per 100-candidate query; inference serializes concurrent requests.
Check queue latency against client timeouts before enabling the endpoint.

The offline public-data trial had case regressions and failed the strict
no-regression gate. An explicitly requested official Smoke is experimental
validation, not promotion or evidence of production readiness.

## Independent protection ablations

With pure `AML_PAIR_RERANK_MODE=model`, two independent flags default to off:

- `AML_FLAG_PAIR_TOP20_PROTECT=1`: preserve baseline Top20 membership and the
  remaining pool membership separately; MiniLM still reorders within each bucket.
  This does not freeze the prefix order or add a Top5 boundary.
- `AML_FLAG_PAIR_EXACT_PROTECT=1`: pin at most 3 qualifying baseline positions
  across the complete scored pool, using the existing exact/phrase/entity/date/
  number signals, filtered query entities and distinct source/content anchors.
  All other positions remain free, including movement across rank 20.

These signals indicate potential evidence, not correctness or factual entailment.
The exact flag reuses existing generic protection qualification; no benchmark
labels, gold answers, question IDs or new query-specific rules are consulted.
Both flags are rejected with pair RRF to keep model scoring and fusion separate.
They can be combined in code, but the initial experiment compares baseline,
pure MiniLM, Top20 only, and exact only, with no combined arm or weight sweep.
Model input, artifact hash, truncation, returned pool and raw logits stay fixed.
Diagnostics add `pair_top20_protect`, `pair_exact_protect`, and `pair_exact_anchor`.
