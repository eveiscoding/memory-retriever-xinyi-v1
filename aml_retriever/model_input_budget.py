"""Model-only limits; lexical recall always receives the original query."""
from contextvars import ContextVar
from contextlib import contextmanager
import logging

QUERY_TOKENS = 256
ROLE_TOKENS = 16
logger = logging.getLogger('aml_retriever.fixed_window')
_context = ContextVar('fixed_window_request', default=None)

def compact(text, tokenizer, limit=QUERY_TOKENS):
    if limit < 1:
        return ''
    if len(tokenizer.encode(text, add_special_tokens=False, truncation=False)) <= limit:
        return text
    offsets = [(int(a), int(b)) for a,b in tokenizer(text, add_special_tokens=False,
        truncation=False, return_offsets_mapping=True)['offset_mapping'] if b>a]
    n = min(len(offsets), limit)
    while n > 0:
        left = (n+1)//2; right = n//2
        candidate = text[:offsets[left-1][1]]
        if right:
            candidate += ' … ' + text[offsets[-right][0]:]
        if len(tokenizer.encode(candidate, add_special_tokens=False, truncation=False)) <= limit:
            return candidate
        n -= 1
    raise ValueError('cannot construct model query')

def model_query(query, options, tokenizer):
    question = compact(query, tokenizer)
    if not options or question != query:
        return question
    combined = query + ' ' + ' '.join(options)
    if len(tokenizer.encode(combined, add_special_tokens=False, truncation=False)) <= QUERY_TOKENS:
        return combined
    # Keep the complete question; only options may use the remaining space.
    rest = ' '.join(options)
    allowance = QUERY_TOKENS-len(tokenizer.encode(query, add_special_tokens=False, truncation=False))-1
    while allowance > 0:
        candidate = query + ' ' + compact(rest, tokenizer, allowance)
        if len(tokenizer.encode(candidate, add_special_tokens=False, truncation=False)) <= QUERY_TOKENS:
            return candidate
        allowance -= 1
    return question

@contextmanager
def request_context(request_id, parent_query):
    state = dict(request_id=request_id, parent_query=parent_query, degraded=False)
    token = _context.set(state)
    try:
        yield state
    finally:
        _context.reset(token)

def event(stage, strategy, error=None, **counts):
    state = _context.get()
    if state is not None and strategy.endswith('fallback'):
        state['degraded'] = True
    # Never serialize exception messages, identifiers, query, options or evidence.
    emit = logger.warning if strategy.endswith('fallback') else logger.info
    emit('fixed_window_event request=%s stage=%s strategy=%s type=%s counts=%s',
        state['request_id'] if state else 'none', stage, strategy,
        type(error).__name__ if error else 'none', counts)
