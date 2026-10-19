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
