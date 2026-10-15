# Retrieval improvement roadmap

This roadmap is intentionally retrieval-only. The official interface consumes
`Search` results and does not require this repository to generate an answer.
Answer-model experiments may be used as one-off diagnostics, but they are not a
release gate and must not replace retrieval metrics.

## Branch discipline

The core and independent benchmark repositories use the same numbered branch
prefix when a phase touches them both. Every phase starts from the committed,
tested result of the previous phase; phases are not developed as parallel
sibling branches.

| Phase | Core branch | Benchmark branch | Primary outcome |
| --- | --- | --- | --- |
| 01 | `codex/01-retrieval-diagnostics` | `codex/01-retrieval-diagnostics` | Retrieval-only diagnostics, semantic LoCoMo slices, and frozen baselines |
| 02 | `codex/02-hybrid-retrieval` | `codex/02-hybrid-retrieval` | Add a dense candidate channel and fuse it with FTS/features without replacing exact retrieval |
| 03 | `codex/03-recall-preserving-rerank` | `codex/03-recall-preserving-rerank` | Reorder a protected candidate pool with a semantic reranker while preserving candidate recall |
| 04 | `codex/04-hierarchical-session-retrieval` | `codex/04-hierarchical-session-retrieval` | Session-first retrieval, auxiliary summary/fact keys, and turn-level provenance recovery |
| 05 | `codex/05-preference-profile-memory` | `codex/05-preference-profile-memory` | Typed preference/profile evidence with source role, confidence, validity, and supersession |
| 06 | `codex/06-bitemporal-state-memory` | `codex/06-bitemporal-state-memory` | Event time, record time, validity intervals, and current-versus-historical query handling |
| 07 | `codex/07-multihop-memory-links` | `codex/07-multihop-memory-links` | Query decomposition and bounded relation expansion for multi-session and implicit-connection retrieval |

If a phase needs no change in one repository, that repository still receives a
small phase branch containing the updated experiment pointer or documentation,
so the ancestry and paired revisions stay explicit.

## Phase 01 — retrieval diagnostics

Work:

- keep the public benchmark reports retrieval-only;
- add Complete@5/100, last required evidence rank, gold-item reciprocal rank,
  content duplication, and source-overlap diagnostics;
- expose semantic LoCoMo scenarios while retaining the raw category number;
- explicitly separate adversarial evidence retrieval from abstention quality;
- freeze clean, locked baselines before changing retrieval behavior.

Acceptance:

- all existing tests pass and new metric tests cover missing and multi-evidence
  cases;
- the phase changes reporting only, not Add/Search behavior;
- a locked smoke report can be compared with the previous report without
  silently changing the meaning of Recall or MRR.

## Phase 02 — hybrid candidate generation

Work:

- add a pluggable dense embedding backend and persistent embedding storage;
- retrieve FTS, deterministic-feature, and dense candidates independently;
- fuse channels with configurable RRF and preserve channel/rank provenance;
- keep the current zero-dependency path functional when no model is configured.

Acceptance:

- improve LongMemEval multi-session and preference Recall@20 or Recall@100;
- do not regress single-session user/assistant Complete@20;
- record Add/Search p50/p95, index size, model identity, and model hash.

## Phase 03 — recall-preserving semantic rerank

Work:

- rerank only a protected Top-N candidate pool;
- retain an escape quota for exact numeric/date/entity matches;
- make reranking optional and expose pre/post ranks in diagnostics;
- cache deterministic query-candidate scores for reproducible benchmarks.

Acceptance:

- candidate Recall@100 is identical by construction;
- improve preference and multi-session MRR/nDCG without reducing Recall@20;
- remain within the recorded Search latency budget.

## Phase 04 — hierarchical session retrieval

Work:

- add session summaries, keyphrases, user facts, and event keys as auxiliary
  indexes; never replace raw messages;
- rank sessions first, then retrieve original turns inside selected sessions;
- apply per-session diversity so duplicate windows cannot consume the evidence
  budget;
- preserve source message IDs through every derived view.

Acceptance:

- improve multi-session Complete@20 and last-required-evidence rank;
- reduce source overlap in Top-20;
- preserve exact-fact and temporal/update regression gates.

## Phase 05 — preference/profile memory

Work:

- represent subject, preference target, polarity, explicit/implicit source,
  source role, confidence, validity interval, and provenance;
- favor direct first-person user evidence without hiding assistant or third-party
  evidence;
- support changed preferences through explicit supersession.

Acceptance:

- improve LongMemEval and ConvoMem preference MRR/nDCG;
- no regression on user/assistant facts or knowledge-update slices;
- validate weights on public slices before changing defaults.

## Phase 06 — bitemporal state memory

Work:

- store event time separately from ingestion/record time;
- represent `valid_from`, `valid_until`, and `supersedes` links;
- classify current-state, historical-state, and temporal-order queries;
- keep obsolete evidence retrievable for historical questions.

Acceptance:

- improve temporal and changing-fact ranking;
- reduce outdated/conflicting evidence above the correct current state;
- recover historical values without destructive deletion.

## Phase 07 — multi-hop memory links

Work:

- decompose eligible queries into a bounded number of subqueries;
- link candidates by normalized entities, events, time, and session;
- perform bounded one/two-hop expansion and return the smallest evidence set
  covering the subqueries;
- keep graph-derived results tied to immutable source messages.

Acceptance:

- improve LongMemEval multi-session, LoCoMo multi-hop, and ConvoMem implicit
  connections;
- cap expansion work and report latency/candidate growth;
- preserve user isolation and provenance checks.

## Common gates

Every phase must use a clean checkout, fixed dataset hashes, a run lock, a fresh
SQLite database, a loopback-only endpoint, and the same benchmark revision for
before/after comparison. Report improvements, regressions, unchanged metrics,
latency, storage, and unscored rows. A phase is not promoted merely because one
headline metric rises.

Mac development runs use sampled suites. In particular, LongMemEval uses the
six-question local sample by default; the 120-question balanced suite and any
larger run execute only on the server after transfer and revision verification.
