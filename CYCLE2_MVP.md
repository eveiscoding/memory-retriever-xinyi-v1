# Cycle 2 textual-track MVP

This branch adapts `dlxeva/flowgrid-aml-retriever` for the second Agent Memory
Leaderboard textual track, including Streaming evaluation.

## Submission endpoints

- `POST /add` — synchronous; committed memory is searchable before HTTP 200.
- `POST /search` — returns memory evidence only and never more than `top_k`.
- `GET /health` — unauthenticated 2xx health probe.
- Authentication — Bearer token in production.

## Production changes from upstream

- Fail-closed authentication configuration and constant-time key comparison.
- Payload-hash validation for idempotent Add; conflicting reuse returns HTTP 409.
- Bounded in-flight requests with HTTP 429 backpressure.
- Thirty-day per-user retention cleanup.
- Admin deletion and stats endpoints disabled by default.
- Container runs unprivileged, binds only to loopback, and is exposed by Caddy.
- Cycle 2 incremental Streaming regression coverage.

## Attribution

Based on [dlxeva/flowgrid-aml-retriever](https://github.com/dlxeva/flowgrid-aml-retriever),
licensed under MIT. The Cycle 2 deployment and safety changes are maintained in
this branch; the original copyright and license are preserved in `LICENSE`.

## Remaining score work

The MVP remains deterministic SQLite FTS5 retrieval. Its principal known gap is
semantically distant paraphrase recall. A model-backed Add enrichment stage and
official-evaluation tuning should be treated as the next iteration, not silently
enabled without a configured model key and cost controls.
