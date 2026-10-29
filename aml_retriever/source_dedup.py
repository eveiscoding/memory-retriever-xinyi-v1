"""Opt-in stable deferral of fully covered source evidence.

No semantic similarity, role filtering, time preference, model call, deletion,
or score change. Both lineage subset and full text containment are required.
"""
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


def source_aware_order(records, *, user_id):
    """Return (all original records in new order, per-ID decision metadata).

    Caller supplies one tenant's existing ranked pool. Earlier retained records
    are the only covering witnesses. Unknown lineage/text is retained. A union
    of partial records is insufficient: one witness must contain the complete
    whitespace-normalized body, preserving cross-message context as well.
    """
    records = list(records)
    ids = [r.get('id') for r in records]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('source ordering requires unique document IDs')
    if any(r.get('user_id', user_id) != user_id for r in records):
        raise ValueError('source ordering requires a single tenant')
    kept, deferred, witnesses, audit = [], [], [], {}
    for rank, record in enumerate(records, 1):
        sources, text = _sources(record), _text(record.get('content'))
        witness = next((w for w, ws, wt in witnesses
                        if sources and sources <= ws and text and text in wt), None)
        decision = dict(status='retained', original_rank=rank)
        if witness is None:
            kept.append(record)
            if sources and text:
                witnesses.append((record['id'], sources, text))
        else:
            deferred.append(record)
            decision.update(status='deferred', covered_by=witness,
                            reason='source_subset_and_full_text_containment')
        audit[record['id']] = decision
    return kept + deferred, audit
