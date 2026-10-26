# Indexed FTS writes (isolated candidate)

Baseline: `9fc9c6487b36fe1586ba4a1df73eada0aa8e943e`.
This is not the frozen Full deployment and must not be pointed at its database.

FTS5 stores `doc_id` as UNINDEXED. The old replacement deletes by that column,
scanning the growing FTS table for each message/view. This candidate adds a
normal B-tree mapping from doc_id/user_id to FTS rowid, then deletes by rowid.
It retains the existing delete-then-insert rowid behavior, tokenizer, documents,
ranking, synchronous commit, write serialization and idempotency contract.
No background write is acknowledged as complete.

Mapping creation/backfill is one atomic migration under BEGIN IMMEDIATE after
schema setup. It preserves existing FTS rowids and duplicate legacy doc_ids.
Replacement, scoped deletion and TTL cleanup maintain both tables in their
existing write transaction; purge_all clears both. Migration takes an initial
scan and write lock, so measure its duration separately from steady-state Add.

Only one software version may write a database. Do not downgrade a migrated
database to an old writer and later reopen it with this candidate: the old
writer does not maintain the mapping. Use separate databases for the frozen
baseline and candidate; production migration/rollback requires separate review.

Targeted local fixtures verify mapping, migration, rollback, deletion and query
plan. No local large-data benchmark or full test suite is authorized by default.
Server full tests and bounded synthetic scaling/concurrency measurements remain
required before claiming performance gains or pushing. No retrieval quality
gain or official timeout root cause is established by this implementation.

Server-only paired check (new output directory, no production volume):

```text
python3 tools/verify_indexed_fts_writes.py --baseline-core /path/to/frozen/checkout \
  --output /path/to/new/synthetic-run --sizes 20000,100000
```

The driver creates real synthetic message/view seeds through Add, drops only
their mapping table to simulate a legacy database, and clones identical bytes
for two separate import-locked processes. It records migration separately,
30 same-content index replacements with SQLite VM progress ticks, 20 serial
and 160 concurrent (16 workers) 20-message loopback HTTP Adds plus immediate
Search checks, retry idempotency, isolation, deletion, reopen, SQLite/FTS/map
integrity, logical table hashes and exact 20-query retrieval fingerprints.
Wall-clock write timestamps and score reference time are fixed only for these
synthetic comparisons; production clock behavior is unchanged. No external
model is invoked. Each worker times out at 600 seconds. Use host CPU/memory
limits and preserve failed output directories. Passing these bounded tests is
not proof of arbitrary long-run stability or the platform error's root cause.
