"""Experimental numeric fact support on a frozen pool; no gold, Add or cache."""
from __future__ import annotations
from dataclasses import dataclass
import math
import re

_NUMBER = r"(?:\d+(?:[.,]\d+)*|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|hundred|thousand)"
_NUM = re.compile(r"\b" + _NUMBER + r"\b", re.I)
_DURATION = re.compile(r"\b" + _NUMBER + r"\s+(?:days?|weeks?|months?|years?|hours?|nights?)\b", re.I)
_DATE_RANGE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*(?:to|through|until|-)\s*(?:[A-Z][a-z]+\s+)?\d{1,2}\b", re.I)
_MONEY = re.compile(r"[$€£]\s*" + _NUMBER + r"\b|\b" + _NUMBER + r"\s+(?:dollars?|euros?|pounds?|yen)\b", re.I)
_PERSONAL = re.compile(r"\b(?:i|my|our|we|you|your)\b", re.I)
_ASSERTION = re.compile(r"\b(?:am|is|are|was|were|have|has|had|went|traveled|travelled|spent|spend|paid|pay|cost|costs|stayed|stay|visited|lived|been|collected|collecting|own|owns|bought|purchased|booked|occupy|occupies|hold|holds|scored|score|tried|made|total)\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:if|would|could|should|may|might|can|recommend|recommendations|suggest|suggestions|plan|planning|considering|will|never|not|no longer)\b", re.I)
_STOP = set("what which how many much long more total number percentage percent ratio highest current score cost price spend spent paid purchase purchased hold held occupy did do does have has had i my our we you your in on at to of the a an for from and or compared than since been be is are was were it me this that so far per by with about actually now time times day days week weeks month months year years hour hours night nights".split())
_DOMAINS = (
    frozenset("accommodation accommodations lodging hotel hotels hostel hostels resort resorts rent rental".split()),
    frozenset("food meal meals breakfast lunch dinner restaurant restaurants dining".split()),
    frozenset("flight flights airfare ticket tickets train bus taxi transport transportation".split()),
)

def _tokens(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:-3] + "y" if w.endswith("ies") else w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}

@dataclass(frozen=True)
class Need:
    kind: str
    anchors: frozenset[str]
    names: frozenset[str]
    domains: tuple[frozenset[str], ...]
    text: str

def query_needs(query):
    """Conservative English intent; split conjuncts without answer values."""
    q = query.strip()
    if len(q) > 2048:
        return ()
    if re.search(r"\b(?:how long|how many (?:days|weeks|months|years|hours|nights)|number of (?:days|weeks|months|years|hours|nights))\b", q, re.I):
        kind = "duration"
    elif re.search(r"\b(?:cost|price|paid|spend|spent|expensive|accommodations?)\b", q, re.I):
        kind = "money"
    elif re.search(r"\b(?:how many|number of|percentage|percent|ratio|highest score)\b", q, re.I):
        kind = "quantity"
    else:
        return ()
    needs = []
    global_domains = tuple(frozenset(_tokens(" ".join(d))) for d in _DOMAINS if _tokens(q) & _tokens(" ".join(d)))
    for part in re.split(r"\b(?:and|compared to|versus|vs)\b", q, flags=re.I):
        anchors = _tokens(" ".join(w for w in re.findall(r"[A-Za-z]+", part) if w.lower() not in _STOP))
        if not anchors:
            continue
        names = _tokens(" ".join(w for w in re.findall(r"\b[A-Z][a-z]+\b", part) if w.lower() not in _STOP))
        domains = tuple(frozenset(_tokens(" ".join(d))) for d in _DOMAINS if _tokens(part) & _tokens(" ".join(d))) if kind == "money" else ()
        if kind == "money" and not domains:
            domains = global_domains
        needs.append(Need(kind, frozenset(anchors), frozenset(names), domains, part.strip()))
    return tuple(needs[:8])

def _clauses(content, max_chars):
    # Decimal/currency values stay intact. Never embed just the head of an
    # eligible overlong clause. Truncation and budgets are diagnostic, not hidden.
    text = content[:max_chars]
    if len(content) > max_chars:
        end = max(text.rfind("\n"), text.rfind(". "))
        text = text[:max(0, end)]
    for index, raw in enumerate(re.split(r"\n+|(?<=[.!?;])\s+|\b[Bb]y the way[, :]\s*", text)):
        clause = re.sub(r"^\s*(?:user|assistant):\s*", "", raw).strip()
        if clause and len(clause) <= 800:
            yield index, clause

def _eligible(need, clause):
    uncertainty_text = re.sub(r"\bMay(?=\s+\d)", "MONTH", clause)
    if "?" in clause or _UNCERTAIN.search(uncertainty_text):
        return False
    if not _PERSONAL.search(clause) or not _ASSERTION.search(clause):
        return False
    words = _tokens(clause)
    other_subject = re.match(r"(?:my|your|our)\s+(friend|sister|brother|mother|father|partner|spouse|colleague|cousin)\b", clause, re.I)
    if other_subject and not (_tokens(other_subject.group(1)) & _tokens(need.text)):
        return False
    if need.names and not need.names.issubset(words):
        return False
    if need.domains and not all(words & domain for domain in need.domains):
        return False
    if need.kind == "money":
        return bool(_MONEY.search(clause))
    if need.kind == "duration":
        return bool(_DURATION.search(clause) or _DATE_RANGE.search(clause))
    count_text = _DATE_RANGE.sub("", _DURATION.sub("", _MONEY.sub("", clause)))
    return bool(_NUM.search(count_text))

def validate_policy(mode, weight, k, max_chars, per_doc, max_total, min_similarity):
    if mode not in {"rules", "embedding"}:
        raise ValueError("fact support mode must be rules or embedding")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("fact support weight must be finite in [0,1]")
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0:
        raise ValueError("fact support k must be positive and finite")
    for value, upper in [(max_chars, 131072), (per_doc, 16), (max_total, 1600)]:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError("invalid fact support work budget")
    if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)) or not math.isfinite(min_similarity) or not 0 <= min_similarity < 1:
        raise ValueError("fact support min similarity must be finite in [0,1)")

def _vectors(backend, texts):
    raw = backend.embed(texts)
    if len(raw) != len(texts):
        raise ValueError("invalid fact embedding count")
    dimension = int(backend.dimension)
    output = []
    for vector in raw:
        if len(vector) != dimension or any(isinstance(v, bool) for v in vector):
            raise ValueError("invalid fact embedding dimension/type")
        values = [float(v) for v in vector]
        norm = math.sqrt(sum(v*v for v in values))
        if not all(math.isfinite(v) for v in values) or not math.isfinite(norm) or norm == 0:
            raise ValueError("invalid fact embedding values")
        output.append([v / norm for v in values])
    return output

def fact_support_order(records, query, *, mode="rules", backend=None, weight=.1, k=60.,
                       max_chars=32768, per_doc=4, max_total=400, min_similarity=.25):
    """Atomic fixed-pool rerank. Constant/missing support is a true no-op.

    Diagnostic score fields remain unchanged. No cross-request text cache.
    Optional backend sees only eligible clauses from this request's pool.
    """
    validate_policy(mode, weight, k, max_chars, per_doc, max_total, min_similarity)
    if not records or weight == 0:
        return records
    needs = query_needs(query)
    if not needs:
        return records
    if mode == "embedding" and backend is None:
        raise ValueError("fact support embedding mode requires a backend")
    per_record = []
    for record in records:
        candidates = []
        for index, clause in _clauses(record.get("content") or "", max_chars):
            for ni, need in enumerate(needs):
                if not _eligible(need, clause):
                    continue
                overlap = len(need.anchors & _tokens(clause)) / len(need.anchors)
                candidates.append((overlap, index, ni, clause))
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        per_record.append(candidates[:per_doc])
    # Round-robin prevents one long early document consuming the whole budget.
    chosen = []
    for slot in range(per_doc):
        for ri, candidates in enumerate(per_record):
            if slot < len(candidates) and len(chosen) < max_total:
                chosen.append((ri, candidates[slot]))
    if not chosen:
        return records
    vectors = None
    offsets = {}
    if mode == "embedding":
        texts = list(dict.fromkeys([n.text for n in needs] + [c[3] for _, c in chosen]))
        offsets = {text: i for i, text in enumerate(texts)}
        vectors = _vectors(backend, texts)
    scores = [0.] * len(records)
    sentence_index = [None] * len(records)
    for ri, (lexical, index, ni, clause) in chosen:
        support = lexical
        if vectors is not None:
            a, b = vectors[offsets[needs[ni].text]], vectors[offsets[clause]]
            cosine = max(-1., min(1., sum(x*y for x, y in zip(a, b))))
            support = max(0., (cosine - min_similarity) / (1. - min_similarity))
        if support > scores[ri]:
            scores[ri], sentence_index[ri] = support, index
    if max(scores) == min(scores):
        return records
    support_rank = {value: 1 + sum(s > value for s in scores) for value in set(scores)}
    fused = [(1-weight)/(k+i+1) + weight*s/(k+support_rank[s]) for i, s in enumerate(scores)]
    order = sorted(range(len(records)), key=lambda i: (-fused[i], i))
    output = []
    for rank, i in enumerate(order, 1):
        record = dict(records[i])
        record["flags"] = list(record.get("flags", [])) + ["fact_support_" + mode]
        record["channel_ranks"] = dict(record.get("channel_ranks", {}), fact_before=i+1, fact_after=rank)
        record["fact_support_trace"] = dict(support=round(scores[i], 6), clause_index=sentence_index[i],
            mode=mode, scored_clauses=len(chosen), scan_truncated=len(record.get("content") or "") > max_chars,
            backend_identity=str(backend.identity) if vectors is not None else None,
            backend_hash=str(backend.model_hash) if vectors is not None else None)
        output.append(record)
    return output
