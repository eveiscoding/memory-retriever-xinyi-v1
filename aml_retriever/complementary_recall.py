"""Bounded keyword/entity-graph/explicit-time/optional-semantic fact recall.

Hindsight-inspired independent retrieval and rank fusion, not a Hindsight port.
No entity extraction, natural-language time inference, or model installation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
import unicodedata

from . import features, temporal_facts


def validate_config(config):
    for value, lo, hi in ((config.fact_recall_scan_limit, 1, 10000),
                         (config.fact_recall_channel_limit, 1, 400),
                         (config.fact_recall_graph_hops, 1, 2),
                         (config.fact_recall_embed_chars, 1, 1000000),
                         (config.fact_recall_rrf_k, 1, 10000)):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError("invalid complementary recall budget")
    if type(config.fact_recall_weight) not in (int, float) or not math.isfinite(config.fact_recall_weight) or not 0 <= config.fact_recall_weight <= 1:
        raise ValueError("invalid complementary recall weight")


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def init_schema(con):
    con.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
            text, fact_id UNINDEXED, user_id UNINDEXED);
        CREATE TABLE IF NOT EXISTS fact_entities(
            user_id TEXT NOT NULL, fact_id TEXT NOT NULL, entity TEXT NOT NULL,
            message_id TEXT NOT NULL, quote TEXT NOT NULL, recorded_at INTEGER NOT NULL,
            PRIMARY KEY(user_id,fact_id,entity,message_id,quote)
        );
        CREATE INDEX IF NOT EXISTS idx_fact_entities_user ON fact_entities(user_id,fact_id);
    """)


def link_entity(con, *, user_id, fact_id, entity, message_id, quote):
    if not all(isinstance(v, str) and v.strip() for v in (user_id, fact_id, entity, message_id, quote)):
        raise ValueError("entity link requires nonempty strings")
    if len(entity) > 128 or len(quote) > 2048:
        raise ValueError("entity annotation exceeds limit")
    label = normalize(entity)
    if label not in normalize(quote):
        raise ValueError("entity must appear in its source quote; aliases need separate resolution")
    source = con.execute("SELECT quote FROM fact_sources WHERE user_id=? AND fact_id=? AND message_id=?",
                         (user_id, fact_id, message_id)).fetchall()
    if not any(quote in row["quote"] for row in source):
        raise ValueError("entity quote must be anchored to this fact's cited source")
    count = con.execute("SELECT COUNT(*) FROM fact_entities WHERE user_id=? AND fact_id=?",
                        (user_id, fact_id)).fetchone()[0]
    exists = con.execute("SELECT 1 FROM fact_entities WHERE user_id=? AND fact_id=? AND entity=? AND message_id=? AND quote=?",
                         (user_id, fact_id, label, message_id, quote)).fetchone()
    if count >= 16 and not exists:
        raise ValueError("entity links per fact exceed limit")
    con.execute("INSERT OR IGNORE INTO fact_entities VALUES(?,?,?,?,?,?)",
                (user_id, fact_id, label, message_id, quote, temporal_facts.now_ms()))


def date_point(query):
    dates = re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", query)
    if not dates:
        return None
    if len(set(dates)) != 1:
        raise ValueError("multiple dates need an explicit temporal query, not a guessed window")
    return int(datetime.strptime(dates[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _contains_entity(query, entity):
    if re.search(r"[\u3400-\u9fff]", entity):
        return entity in normalize(query)  # Chinese labels have no word spaces.
    return re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", normalize(query)) is not None


def _knowledge_ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def recall(con, *, user_id, query, config, backend=None, valid_at=None, known_at=None, limit=50, flags=None):
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id is required")
    if not isinstance(query, str) or not query.strip() or len(query) > 16000:
        raise ValueError("query must be a bounded nonempty string")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("recall limit must be between 1 and 100")
    flags = config.flags if flags is None else flags
    point = temporal_facts.timestamp(valid_at) if valid_at is not None else (
        date_point(query) if flags.get("fact_recall_temporal", True) else None)
    if (point is not None or known_at is not None) and not flags.get("fact_versions", False):
        raise ValueError("temporal recall requires fact_versions")
    known = temporal_facts.timestamp(temporal_facts.now_ms() if known_at is None else known_at)
    scan = config.fact_recall_scan_limit
    channel_limit = config.fact_recall_channel_limit
    rows = con.execute("SELECT * FROM facts WHERE user_id=? ORDER BY fact_id LIMIT ?", (user_id, scan + 1)).fetchall()
    truncated = len(rows) > scan
    facts = {r["fact_id"]: dict(r) for r in rows[:scan]}
    if flags.get("fact_versions", False):
        ids = list(facts)
        versioned, visible = set(), set()
        when = temporal_facts.now_ms() if point is None else point
        # Eligibility is evaluated for the complete bounded candidate universe,
        # not after a result-limit that could let stale versions consume slots.
        for offset in range(0, len(ids), 400):
            part = ids[offset:offset + 400]
            marks = ",".join("?" for _ in part)
            for r in con.execute(f"SELECT * FROM fact_versions WHERE user_id=? AND fact_id IN ({marks})",
                                 (user_id, *part)):
                if r["recorded_at"] <= known:
                    versioned.add(r["fact_id"])
                if (r["recorded_at"] <= known and (r["retired_at"] is None or known < r["retired_at"])
                        and r["valid_at"] is not None and r["valid_at"] <= when
                        and (r["invalid_at"] is None or when < r["invalid_at"])):
                    visible.add(r["fact_id"])
        facts = {fid:f for fid,f in facts.items() if fid in visible or (
            point is None and fid not in versioned and _knowledge_ms(f["created_at"]) <= known)}
    orders, status = {}, {}
    tokens = features.query_tokens(query, config.max_query_tokens)
    for name in ("keyword", "graph", "temporal", "semantic"):
        status[name] = "disabled" if not flags.get("fact_recall_" + name, True) else "empty"
        orders[name] = []
    if status["keyword"] != "disabled" and tokens and facts:
        expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        # Restrict before ranking/limiting, preserving true FTS5 BM25 and tenant
        # isolation. Separate FTS never changes the baseline corpus statistics.
        ranked = []
        ids = list(facts)
        for offset in range(0, len(ids), 400):
            part = ids[offset:offset + 400]
            marks = ",".join("?" for _ in part)
            ranked.extend((r["fact_id"], r["rank"]) for r in con.execute(
                f"SELECT fact_id,bm25(fact_fts) AS rank FROM fact_fts WHERE fact_fts MATCH ? "
                f"AND user_id=? AND fact_id IN ({marks})", (expr, user_id, *part)))
        orders["keyword"] = [fid for fid,_ in sorted(ranked, key=lambda r:(r[1],r[0]))[:channel_limit]]
    if status["graph"] != "disabled" and facts:
        entity_facts, fact_entities = {}, {}
        ids = list(facts)
        for offset in range(0, len(ids), 400):
            part = ids[offset:offset + 400]
            marks = ",".join("?" for _ in part)
            for r in con.execute(f"SELECT fact_id,entity FROM fact_entities WHERE user_id=? AND recorded_at<=? AND fact_id IN ({marks})",
                                 (user_id, known, *part)):
                entity_facts.setdefault(r["entity"], set()).add(r["fact_id"])
                fact_entities.setdefault(r["fact_id"], set()).add(r["entity"])
        seeds = {fid for e,fids in entity_facts.items() if _contains_entity(query, e) for fid in fids}
        reached = {fid:1 for fid in seeds}
        frontier = seeds
        for depth in range(2, config.fact_recall_graph_hops + 1):
            frontier = {other for fid in frontier for e in fact_entities.get(fid, ())
                        for other in entity_facts[e] if other not in reached}
            reached.update({fid:depth for fid in frontier})
        orders["graph"] = sorted(reached, key=lambda fid:(reached[fid],fid))[:channel_limit]
    if status["temporal"] != "disabled":
        if point is None:
            status["temporal"] = "not_requested"
        else:
            # Point-in-time validity, not lexical matching or inferred dates.
            orders["temporal"] = sorted(facts)[:channel_limit]
    if status["semantic"] != "disabled":
        if backend is None:
            status["semantic"] = "unavailable"
        elif facts:
            ids = sorted(facts)
            texts = [query] + [facts[fid]["text"] for fid in ids]
            if sum(map(len, texts)) > config.fact_recall_embed_chars:
                status["semantic"] = "input_limit"
            else:
                try:
                    vectors = backend.embed(texts)
                    if len(vectors) != len(texts):
                        raise ValueError("embedding count")
                    if any(len(v) != int(backend.dimension) or any(type(x) not in (int, float)
                            or not math.isfinite(x) for x in v) for v in vectors):
                        raise ValueError("embedding shape/values")
                    def cosine(a, b):
                        norm = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
                        value = sum(x*y for x,y in zip(a,b)) / norm if norm else 0.0
                        if not math.isfinite(value):
                            raise ValueError("embedding magnitude")
                        return value
                    scored = [(fid, cosine(vectors[0], v)) for fid,v in zip(ids, vectors[1:])]
                    orders["semantic"] = [fid for fid,s in sorted(scored, key=lambda r:(-r[1],r[0]))
                                           if s > 0][:channel_limit]
                except Exception:
                    status["semantic"] = "failed"  # no provider exception text
    fused, ranks = {}, {}
    for name, ids in orders.items():
        if ids:
            status[name] = "ok"
        for rank, fid in enumerate(ids, 1):
            fused[fid] = fused.get(fid, 0.0) + 1.0 / (config.fact_recall_rrf_k + rank)
            ranks.setdefault(fid, {})[name] = rank
    result = []
    # System-time API precision is milliseconds; include the entire final ms,
    # while source added_at retains microseconds. Truncating this cutoff would
    # wrongly hide fresh sources recorded in the same millisecond as recall.
    known_iso = datetime.fromtimestamp(known / 1000, tz=timezone.utc).replace(
        microsecond=(known % 1000) * 1000 + 999).isoformat()
    for fid in sorted(fused, key=lambda fid:(-fused[fid],fid))[:limit]:
        item = dict(facts[fid], score=fused[fid], channel_ranks=ranks[fid], derived=True)
        item["sources"] = [dict(r) for r in con.execute(
            "SELECT s.message_id,s.quote FROM fact_sources s JOIN messages m "
            "ON m.id=s.message_id AND m.user_id=s.user_id WHERE s.user_id=? AND s.fact_id=? "
            "AND m.added_at<=? ORDER BY s.message_id,s.quote LIMIT ?",
            (user_id, fid, known_iso, config.max_candidates))]
        if item["sources"]:
            result.append(item)
    return {"results":result, "channels":status, "channel_orders":orders,
            "truncated":truncated, "eligible_facts":len(facts), "valid_at":point, "known_at":known}
