"""Source-aware stable deferral guarded by actual pair-input visibility."""
from __future__ import annotations
import re


def _text(value):
    return re.sub(r'\s+', ' ', value).strip() if isinstance(value, str) else ''


def _sources(record):
    value = record.get('source_ids')
    if not isinstance(value, (list, tuple)) or not value:
        return frozenset()
    if any(not isinstance(x, str) or not x for x in value):
        return frozenset()
    return frozenset(value)


def _span(record, value):
    content = record.get('content')
    if not isinstance(content, str) or not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = value
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (a, b)):
        return None
    return (a, b) if 0 <= a <= b <= len(content) else None


def validate_visible_spans(records, spans):
    """Validate request-local, input-order spans before attaching any metadata."""
    if not isinstance(spans, (list, tuple)) or len(spans) != len(records):
        raise ValueError('pair visible span count mismatch')
    result = {}
    for record, value in zip(records, spans):
        span = _span(record, value)
        if span is None:
            raise ValueError('invalid pair visible span')
        result[record['id']] = span
    return result


def source_aware_order(records, *, user_id, visible_spans=None):
    """One earlier retained witness must cover full text AND its visible text.

    All original records and scores remain. Missing/invalid visibility is never
    interpreted as full visibility. The caller supplies spans for this query and
    model invocation; no gold, text generation, cache or model call is used here.
    """
    records = list(records)
    ids = [r.get('id') for r in records]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('source ordering requires unique document IDs')
    if any(r.get('user_id', user_id) != user_id for r in records):
        raise ValueError('source ordering requires a single tenant')
    visible_spans = visible_spans if isinstance(visible_spans, dict) else {}
    kept, deferred, witnesses, audit = [], [], [], {}
    for rank, record in enumerate(records, 1):
        sources, text = _sources(record), _text(record.get('content'))
        witness = next(((wid, span) for wid, ws, wt, wv, span in witnesses
                        if sources and sources <= ws and text and text in wt and text in wv), None)
        own_span = _span(record, visible_spans.get(record['id']))
        decision = dict(status='retained', original_rank=rank,
                        visibility_known=own_span is not None)
        if witness is None:
            kept.append(record)
            if sources and text and own_span is not None:
                a, b = own_span
                visible = _text(record['content'][a:b])
                if visible:
                    witnesses.append((record['id'], sources, text, visible, own_span))
        else:
            deferred.append(record)
            decision.update(status='deferred', covered_by=witness[0],
                            witness_visible_span=list(witness[1]),
                            reason='source_subset_and_full_text_visible_containment')
        audit[record['id']] = decision
    return kept + deferred, audit
