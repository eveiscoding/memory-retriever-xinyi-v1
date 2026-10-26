"""Server-only bounded synthetic paired write verification; never opens official DBs.

Run from the candidate checkout with --baseline-core and a NEW --output directory.
Each arm is a fresh subprocess importing its own exact checkout, with a cloned
synthetic legacy seed DB. No external models or network except loopback HTTP.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from unittest.mock import patch

FIXED_NOW = "2026-10-01T00:00:00+00:00"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def distribution(values):
    ordered = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[max(0, math.ceil(len(values) * .95) - 1)],
            "max_ms": max(values)}


def checks(db):
    with db.connection() as con:
        assert con.execute("PRAGMA quick_check").fetchall()[0][0] == "ok"
        con.execute("INSERT INTO fts(fts) VALUES('integrity-check')")
        con.commit()
        has_map = con.execute("SELECT 1 FROM sqlite_master WHERE name='fts_doc_rows'").fetchone()
        if has_map:
            assert con.execute("SELECT rowid,doc_id,user_id FROM fts EXCEPT "
                               "SELECT fts_rowid,doc_id,user_id FROM fts_doc_rows").fetchone() is None
            assert con.execute("SELECT fts_rowid,doc_id,user_id FROM fts_doc_rows EXCEPT "
                               "SELECT rowid,doc_id,user_id FROM fts").fetchone() is None
    return {"sqlite_quick_check": True, "fts_integrity": True, "map_exact": bool(has_map)}


def logical_state(db):
    result = {}
    for table, columns, order in [
        ("messages", "id,user_id,session_id,seq,role,content,ts_ms,created_at,request_id", "id"),
        ("views", "view_id,user_id,session_id,view_type,content,created_at,source_ids,start_seq,end_seq,content_hash", "view_id"),
        ("requests", "request_id,user_id,session_id,message_ids,payload_hash", "user_id,request_id"),
        ("sessions", "*", "user_id,session_id"),
        ("fts", "text,doc_id,user_id,doc_type", "doc_id,text"),
    ]:
        rows = [tuple(r) for r in db.query(f"SELECT {columns} FROM {table} ORDER BY {order}")]
        result[table] = {"count": len(rows), "sha256": digest(rows)}
    return result


def worker(args):
    from aml_retriever.config import RetrieverConfig
    from aml_retriever.retriever import RetrieverDB
    from aml_retriever.server import RetrieverServer
    from aml_retriever.api import MemoryService
    from tests.test_api_contract import _payload

    root = Path(args.output)
    if args.worker == "seed":
        db = RetrieverDB(RetrieverConfig(db_path=str(root / "seed.db")))
        with patch("aml_retriever.retriever._now_iso", return_value=FIXED_NOW):
            for start in range(0, args.size, 20):
                user = f"seed-{start // 100}"
                db.add(request_id=f"seed-{start}", user_id=user, session_id="s",
                       messages=[{"role": "user", "timestamp": 1700000000000 + i * 1000,
                                  "content": f"synthetic item{i} account{start // 100} topic{i % 7}"}
                                 for i in range(start, min(start + 20, args.size))])
        seed_checks = checks(db)
        seed_state = logical_state(db)
        db.close()
        # This is our synthetic fixture only. Reconstruct the old schema so
        # candidate migration and baseline open receive identical bytes.
        with sqlite3.connect(root / "seed.db") as con:
            con.execute("DROP TABLE fts_doc_rows")
        return {"seed_checks": seed_checks, "state": seed_state}

    path = root / f"{args.worker}.db"
    shutil.copyfile(root / "seed.db", path)
    before = time.perf_counter()
    config = RetrieverConfig(db_path=str(path), host="127.0.0.1", port=0,
                             auth_mode="bearer", api_key="synthetic-local-only")
    service = MemoryService(config)
    db = service.db
    startup_ms = (time.perf_counter() - before) * 1000
    initial_state = logical_state(db)
    fts_initial = digest([tuple(r) for r in db.query("SELECT rowid,* FROM fts ORDER BY rowid")])
    # Isolate row lookup cost: same pre-existing source and unchanged content,
    # same delete/reinsert semantics. Count SQLite VM progress ticks as well
    # as wall time to separate complexity from shared-host timing noise.
    doc = db.query("SELECT id,user_id,content FROM messages ORDER BY id LIMIT 1")[0]
    micro, ticks = [], [0]
    with db.connection() as con:
        con.set_progress_handler(lambda: ticks.__setitem__(0, ticks[0] + 1) or 0, 1000)
        try:
            for _ in range(30):
                started = time.perf_counter()
                db._index_doc(con, doc[0], doc[1], "message", doc[2])
                con.commit()
                micro.append((time.perf_counter() - started) * 1000)
        finally:
            con.set_progress_handler(None, 0)
    micro_ticks = ticks[0]

    server = RetrieverServer(config, service=service, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(route, payload):
        req = urllib.request.Request(url + route, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer synthetic-local-only"})
        with urllib.request.urlopen(req, timeout=120) as response:
            assert response.status == 200
            return json.loads(response.read())

    def add_one(i):
        user = f"load-{i}"
        payload = _payload(request_id=f"load-{i}", user_id=user, session_id="s",
                           contents=tuple(f"loadtoken{i} synthetic detail{j}" for j in range(20)))
        started = time.perf_counter()
        response = post("/add", payload)
        elapsed = (time.perf_counter() - started) * 1000
        assert response["success"]
        # A completed Add must be immediately queryable, without exposing
        # other users. Exercise the public contract at real loopback HTTP.
        found = post("/search", {"query": f"loadtoken{i}", "user_id": user, "top_k": 10})
        assert found["data"]
        assert all(f"loadtoken{i}" in item["content"] for item in found["data"])
        return elapsed

    try:
        with patch("aml_retriever.retriever._now_iso", return_value=FIXED_NOW):
            seq = [add_one(i) for i in range(20)]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:
                concurrent = list(pool.map(add_one, range(20, 180)))
            concurrent_seconds = time.perf_counter() - started
            assert post("/add", _payload(request_id="load-0", user_id="load-0", session_id="s",
                        contents=tuple(f"loadtoken0 synthetic detail{j}" for j in range(20))))["success"]
        assert db.count() == args.size + 3600
        assert not post("/search", {"query": "loadtoken0", "user_id": "unseen", "top_k": 10})["data"]
        result_state = logical_state(db)
        searches = []
        for i in range(20):
            query = f"item{i}" if i < 10 else f"loadtoken{i}"
            user = "seed-0" if i < 10 else f"load-{i}"
            with patch("aml_retriever.retriever.time.time", return_value=1790812800.0):
                searches.append(asdict(db.search(user_id=user, query=query, top_k=20)))
        correctness = checks(db)
        db.delete_user("load-0")
        assert db.count("load-0") == 0
        assert not db.search(user_id="load-0", query="loadtoken0", top_k=10).results
        assert db.count("load-1") == 20
        checks(db)
    finally:
        server.shutdown()
        server.server_close()
        service.close()
    # Persistence and reopen (including no repeated backfill).
    reopened = RetrieverDB(config)
    try:
        assert reopened.count() == args.size + 3580
        final_checks = checks(reopened)
    finally:
        reopened.close()
    return {"startup_ms": startup_ms, "initial_state": initial_state,
            "initial_fts_rowids_sha256": fts_initial,
            "micro": distribution(micro), "micro_vm_ticks_1000": micro_ticks,
            "sequential_add": distribution(seq), "concurrent_add": distribution(concurrent),
            "concurrent_add_and_search_wall_seconds": concurrent_seconds,
            "state": result_state, "search_sha256": digest(searches),
            "checks": correctness, "reopen_checks": final_checks,
            "http_add_count": 181, "http_search_count": 181,
            "errors": 0, "isolation_and_read_after_write": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-core")
    p.add_argument("--output", required=True)
    p.add_argument("--sizes", default="20000,100000")
    p.add_argument("--worker", choices=["seed", "baseline", "candidate"])
    p.add_argument("--size", type=int)
    args = p.parse_args()
    if args.worker:
        result = worker(args)
        Path(args.output, args.worker + ".json").write_text(json.dumps(result, indent=2))
        return
    candidate = Path(__file__).resolve().parents[1]
    baseline = Path(args.baseline_core).resolve()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    versions = {}
    for name, core in [("baseline", baseline), ("candidate", candidate)]:
        assert not subprocess.check_output(["git", "-C", str(core), "status", "--porcelain"]).strip()
        versions[name] = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"], text=True).strip()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AML_", "OPENAI_", "DASHSCOPE_", "PAIR_SCORE_"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    results = {"versions": versions, "model": "none", "synthetic_only": True,
               "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scales": []}
    for size in map(int, args.sizes.split(",")):
        assert 100 <= size <= 100000 and size % 100 == 0
        root = output / str(size)
        root.mkdir()
        for name, core in [("seed", candidate), ("baseline", baseline), ("candidate", candidate)]:
            arm_env = dict(env, PYTHONPATH=str(core))
            with (root / (name + ".log")).open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name,
                                "--size", str(size), "--output", str(root)],
                               cwd=core, env=arm_env, stdout=log, stderr=log, check=True, timeout=600)
        seed, old, new = [json.loads((root / (name + ".json")).read_text())
                          for name in ("seed", "baseline", "candidate")]
        assert old["initial_state"] == new["initial_state"] == seed["state"]
        assert old["initial_fts_rowids_sha256"] == new["initial_fts_rowids_sha256"]
        assert old["state"] == new["state"]
        assert old["search_sha256"] == new["search_sha256"]
        assert new["micro_vm_ticks_1000"] < old["micro_vm_ticks_1000"]
        assert new["micro"]["median_ms"] < old["micro"]["median_ms"]
        assert new["concurrent_add"]["p95_ms"] < old["concurrent_add"]["p95_ms"]
        results["scales"].append({"seed_messages": size, "fts_docs": seed["state"]["fts"]["count"],
                                  "baseline": old, "candidate": new, "paired_checks_pass": True})
        (output / "report.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"size": size, "pass": True, "old_add_p95": old["concurrent_add"]["p95_ms"],
                          "new_add_p95": new["concurrent_add"]["p95_ms"]}), flush=True)


if __name__ == "__main__":
    main()
