# memory-retriever-xinyi-v1

A user-isolated conversation memory service with synchronous Add/Search APIs,
SQLite FTS5 recall, deterministic ranking features, and optional local ONNX
cross-encoder reranking.

[中文说明](README.zh-CN.md) · [API contract](docs/API_CONTRACT.md) ·
[Pair reranking](docs/PAIR_RERANK.md) · [Indexed writes](docs/INDEXED_FTS_WRITES.md)

## Released implementation

This master release integrates the implementation evaluated at
`888a3b9b793a3c6c97314cf60ef5d22fcba929aa`. Subsequent README and license
commits do not change the evaluated runtime code. The frozen reference is
`codex/smoke-20261027-rrf-indexed-parallel2`.

| Recorded official run | Result |
| --- | --- |
| Textual Full task | `teval_e41f7a2ec090ddd1` |
| Run label | `full-20261027-rrf-v2-888a3b9` |
| Completion | 2026-10-30 12:29:55 UTC+8 |
| Status / Overall | Succeeded / **50.62** |
| Evaluation progress | 9602 / 9602 |

This is the recorded result for the pinned code and deployment profile below.
It is not a new evaluation of the documentation commit or a public leaderboard
ranking. The API returns retrieval evidence; answer generation and grading are
performed by the evaluation platform.

## How it works

1. **Add:** validate the request, persist original messages in SQLite, and update
   FTS5 plus deterministic features before acknowledging success. Request IDs are
   idempotent within a user scope.
2. **Views:** retain raw messages alongside sliding windows and session segments.
   Source message IDs link aggregate evidence back to the original messages.
3. **Recall:** combine FTS5 candidates with a bounded deterministic feature scan.
   Exact terms, entities, dates/numbers, relative recency, and guarded update
   signals contribute to ranking. Options can participate in recall.
4. **Rerank:** the evaluated profile scores the first 100 candidates with a local
   MiniLM cross-encoder, then fuses model rank with baseline rank using RRF.
5. **Output:** deduplicate and preserve raw-message coverage, then return at most
   the requested number of evidence records. All retrieval is scoped by user_id.

Indexed FTS writes avoid global document-ID scans. SQLite uses WAL, a read
connection pool, a serialized write lock, and bounded busy retries. The pooled
ONNX backend owns exactly two independent tokenizer/model sessions; each Search
borrows one for its scoring call.

## Local quick start

Requires Python 3.11+ and SQLite with FTS5. The default lexical/deterministic
profile uses Python's standard library; model reranking is opt-in.

```bash
git clone https://github.com/eveiscoding/memory-retriever-xinyi-v1.git
cd memory-retriever-xinyi-v1
python3 -m unittest discover -s tests
python3 scripts/smoke_api.py
python3 -m aml_retriever.cli serve --db ./aml.db --host 127.0.0.1 --port 8080
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8080/add \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","user_id":"demo-user","session_id":"demo-session","messages":[{"role":"user","content":"My preferred meeting time is Tuesday morning."}]}'

curl -sS http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","query":"When do I prefer meetings?","top_k":10}'
```

Add returns `success` and echoes `request_id/user_id/session_id`. Search returns
`{"data":[{"id":"...","content":"..."}]}`, with optional score, timestamp and
provenance. GET `/health` is unauthenticated. Use `AML_AUTH_MODE=bearer` and
`AML_API_KEY` to protect Add/Search. Admin endpoints are disabled by default.
The API schema and validation details are in [API_CONTRACT](docs/API_CONTRACT.md).

## Evaluated ONNX/RRF profile

The Full run enabled the following profile rather than the default model-free
configuration:

| Setting | Value |
| --- | --- |
| Model | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Model snapshot revision | `1427fd652930e4ba29e8149678df786c240d8825` |
| ONNX artifact | `onnx/model_quint8_avx2.onnx`, CPU INT8 AVX2 |
| Query/document pair context | 512 tokens; document side is truncated |
| Rerank pool / fusion | 100 / RRF, weight 0.25, k=60 |
| Inference | Two independent sessions, batch=1, threads=1 per session |
| Dense recall / embedding rerank | Disabled in this profile |
| Exact / Top20 pair protection | Disabled in this profile |
| Platform Add / Search concurrency | 16 / 16 |

Prepare the pinned tokenizer files and ONNX artifact in an absolute local
snapshot directory. The backend loads local files only and hashes model assets;
it does not download a model during startup.

```bash
python3 -m pip install numpy==1.26.4 onnxruntime==1.20.1 transformers==4.48.3 sentencepiece==0.2.0

export AML_PAIR_MODEL_PATH="/absolute/path/to/mmarco-mMiniLMv2-L12-H384-v1"
export AML_PAIR_ONNX_FILE="onnx/model_quint8_avx2.onnx"
export AML_PAIR_BACKEND="aml_retriever.onnx_pair_backend:create_pooled_backend"
export AML_FLAG_PAIR_RERANK=1
export AML_PAIR_RERANK_POOL=100
export AML_PAIR_RERANK_MODE=rrf
export AML_PAIR_RRF_WEIGHT=0.25
export AML_PAIR_RRF_K=60
export AML_PAIR_MAX_LENGTH=512
export AML_PAIR_BATCH_SIZE=1
export AML_PAIR_THREADS=1
export AML_FLAG_VECTOR=0
export AML_FLAG_SEMANTIC_RERANK=0
export AML_FLAG_PAIR_TOP20_PROTECT=0
export AML_FLAG_PAIR_EXACT_PROTECT=0
export AML_MAX_INFLIGHT=32

python3 -m aml_retriever.cli serve --db ./aml-rrf.db --host 127.0.0.1 --port 8080 --quiet
```

Set authentication separately for an externally reachable deployment. The code
supports additional dense and semantic experiments, but those switches are not
part of this Full profile. See [PAIR_RERANK](docs/PAIR_RERANK.md) for fallback,
fusion, and model identity behavior.

## Containers and verification

- `Dockerfile` builds the standard-library service.
- `Dockerfile.onnx` adds pinned CPU ONNX dependencies. Model assets must be
  mounted separately and the profile above must be explicitly configured.
- `Dockerfile.rrf-parallel` preserves the historical frozen dependency image;
  its `flowgrid-rrf-frozen-base:20261022` base must already be available locally.

```bash
python3 -m unittest discover -s tests
python3 -m aml_retriever.cli selfcheck
python3 scripts/smoke_api.py
```

These are functional checks with synthetic fixtures, not official quality
measurements. Tests for optional backends use fixtures/mocks and do not establish
real-model performance. Service data, keys, official inputs, and evaluation
outputs are not included in the repository.

## Limits and data handling

SQLite is a single-node store. Cross-encoder scoring adds latency and is bounded
by the two model owners. The default lexical profile and the evaluated model
profile have different retrieval behavior. Model-visible document text is limited
by the pair context even when a returned evidence record contains more text.
Temporal/update heuristics are ranking signals and do not resolve facts into a
canonical truth store.

Raw messages remain stored with provenance. Deletion and stale-user cleanup are
available; per-run evaluation retention must be audited separately. See
[DATA_LIFECYCLE](docs/DATA_LIFECYCLE.md). Model artifacts have their own upstream
license terms.

## Source and license

Original reference repository: https://github.com/dlxeva/flowgrid-aml-retriever

This repository is distributed under the [MIT License](LICENSE). The license
retains the original copyright notice and includes the copyright for modifications.
