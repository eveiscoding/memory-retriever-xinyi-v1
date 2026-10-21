"""Explicit bitemporal fact versions, inspired by Graphiti. No time inference."""
from __future__ import annotations

import hashlib
import json
import time


def now_ms():
    return time.time_ns() // 1000000


def timestamp(value, *, optional=False):
    if value is None and optional:
        return None
    if type(value) is not int or not -62135596800000 <= value <= 253402300799999:
        raise ValueError("time must be integer Unix milliseconds, not a guessed date")
    return value


def init_schema(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS fact_versions(
            version_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, fact_id TEXT NOT NULL,
            version_key TEXT NOT NULL, source_role TEXT NOT NULL,
            valid_at INTEGER, invalid_at INTEGER,
            recorded_at INTEGER NOT NULL, retired_at INTEGER,
            supersedes TEXT, revision_of TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fact_versions_scope
            ON fact_versions(user_id, version_key, recorded_at, retired_at);
        CREATE INDEX IF NOT EXISTS idx_fact_versions_fact
            ON fact_versions(user_id, fact_id, recorded_at, retired_at);
        CREATE TABLE IF NOT EXISTS fact_version_ops(
            user_id TEXT NOT NULL, operation_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
            version_id TEXT NOT NULL, PRIMARY KEY(user_id, operation_id)
        );
    """)


def _overlap(a, b, c, d):
    # Unknown validity is NOT an unbounded interval and never proves conflict.
    return a is not None and c is not None and (b is None or c < b) and (d is None or a < d)


def record(con, *, user_id, fact_id, operation_id, version_key, valid_at,
           invalid_at=None, supersedes=None):
    for name, value in (("user_id", user_id), ("fact_id", fact_id),
                        ("operation_id", operation_id), ("version_key", version_key)):
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(name + " must be a nonempty bounded string")
    start, end = timestamp(valid_at, optional=True), timestamp(invalid_at, optional=True)
    if end is not None and (start is None or end <= start):
        raise ValueError("valid interval must be known and nonempty [start,end)")
    if supersedes is not None and (not isinstance(supersedes, str) or not supersedes):
        raise ValueError("invalid predecessor")
    data = [fact_id, version_key, start, end, supersedes]
    sha = hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()
    prior_op = con.execute("SELECT * FROM fact_version_ops WHERE user_id=? AND operation_id=?",
                           (user_id, operation_id)).fetchone()
    if prior_op:
        if prior_op["payload_hash"] != sha:
            raise ValueError("version operation id reused with a different payload")
        return dict(con.execute("SELECT * FROM fact_versions WHERE user_id=? AND version_id=?",
                                (user_id, prior_op["version_id"])).fetchone())
    fact = con.execute("SELECT source_role FROM facts WHERE user_id=? AND fact_id=?",
                       (user_id, fact_id)).fetchone()
    if fact is None:
        raise ValueError("fact not found in user scope")
    previous = None
    if supersedes is not None:
        previous = con.execute("SELECT * FROM fact_versions WHERE user_id=? AND version_id=?",
                               (user_id, supersedes)).fetchone()
        if previous is None or previous["retired_at"] is not None:
            raise ValueError("predecessor is not a current version in user scope")
        if previous["version_key"] != version_key or previous["source_role"] != fact["source_role"]:
            raise ValueError("version key or speaker does not match predecessor")
        if start is None or previous["valid_at"] is None or start <= previous["valid_at"]:
            raise ValueError("supersession needs a strictly later explicit valid_at")
    current = con.execute("SELECT * FROM fact_versions WHERE user_id=? AND version_key=? "
                          "AND source_role=? AND retired_at IS NULL",
                          (user_id, version_key, fact["source_role"])).fetchall()
    if any(r["version_id"] != supersedes and _overlap(start, end, r["valid_at"], r["invalid_at"])
           for r in current):
        raise ValueError("overlapping current version requires an explicit predecessor")
    recorded = now_ms()
    latest = con.execute("SELECT MAX(recorded_at) FROM fact_versions WHERE user_id=?", (user_id,)).fetchone()[0]
    if latest is not None and recorded < latest:
        raise ValueError("system clock moved backwards; refusing to rewrite knowledge history")
    version_id = "fv_" + hashlib.sha256(json.dumps([user_id, operation_id]).encode()).hexdigest()
    if previous is not None:
        # Preserve the original assertion for historical knowledge-time queries.
        con.execute("UPDATE fact_versions SET retired_at=? WHERE user_id=? AND version_id=?",
                    (recorded, user_id, supersedes))
        revised_end = min(start, previous["invalid_at"]) if previous["invalid_at"] is not None else start
        con.execute("INSERT INTO fact_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (version_id + "_prior", user_id, previous["fact_id"], version_key, fact["source_role"],
             previous["valid_at"], revised_end, recorded, None, previous["supersedes"], supersedes))
    con.execute("INSERT INTO fact_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (version_id, user_id, fact_id, version_key, fact["source_role"], start, end,
         recorded, None, supersedes, None))
    con.execute("INSERT INTO fact_version_ops VALUES(?,?,?,?)", (user_id, operation_id, sha, version_id))
    return dict(con.execute("SELECT * FROM fact_versions WHERE user_id=? AND version_id=?",
                            (user_id, version_id)).fetchone())


def as_of(con, *, user_id, valid_at, known_at=None, limit=100):
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id is required")
    when = timestamp(valid_at)
    known = timestamp(now_ms() if known_at is None else known_at)
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("version limit must be between 1 and 1000")
    return [dict(r) for r in con.execute(
        "SELECT v.*, f.text, f.kind FROM fact_versions v JOIN facts f "
        "ON f.fact_id=v.fact_id AND f.user_id=v.user_id WHERE v.user_id=? "
        "AND v.recorded_at<=? AND (v.retired_at IS NULL OR v.retired_at>?) "
        "AND v.valid_at IS NOT NULL AND v.valid_at<=? "
        "AND (v.invalid_at IS NULL OR v.invalid_at>?) "
        "ORDER BY v.version_key,v.source_role,v.valid_at,v.version_id LIMIT ?",
        (user_id, known, known, when, when, limit))]
