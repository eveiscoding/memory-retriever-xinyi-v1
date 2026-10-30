"""Explicit fixed-window + 8K source-evidence HTTP service.

The default CLI/service remains unchanged. Official data contains one item whose
content is the complete budgeted rendering used by the offline consumer.
"""
import hashlib
import threading
import uuid
from .model_input_budget import compact, model_query, request_context, event, logger, ROLE_TOKENS
from .api import MemoryService, validate_search_payload, ApiError
from .config import RetrieverConfig
from .reliable_parent_window import ReliableParentWindowRetriever as ParentWindowRetriever
from . import source_evidence_output as output

MODEL_HASH = '781f3f03fe04bf180ba0d912b6ee7f0177cf793176b5a317a8a5e0e8e7036559'
PROFILE = 'fixed-parent-384-64-max-source-8k-query256-v2'

class FixedWindowService(MemoryService):
    def __init__(self, config, db=None):
        config.validate()
        if (not config.flags.get('pair_rerank') or
            not config.flags.get('source_aware_dedup') or
            config.flags.get('vector') or config.flags.get('semantic_rerank') or
            config.flags.get('pair_top20_protect') or config.flags.get('pair_exact_protect') or
            config.pair_rerank_pool != 100 or config.pair_rerank_mode != 'rrf' or
            config.pair_rrf_weight != .25 or config.pair_rrf_k != 60):
            raise ValueError('fixed-window service requires the frozen retrieval profile')
        if db is None:
            from .onnx_pair_backend import OnnxPairBackend
            backend = OnnxPairBackend()
            if (backend.model_hash != MODEL_HASH or backend.batch_size != 1 or
                    backend.max_length != 512):
                raise ValueError('fixed-window model/profile mismatch')
            db = ParentWindowRetriever(config, pair_backend=backend)
        if not isinstance(db, ParentWindowRetriever):
            raise ValueError('fixed-window retriever required')
        super().__init__(config, db)
        self._request_lock = threading.RLock()

    def official_add(self, payload):
        with self._request_lock:
            return super().official_add(payload)

    def official_search(self, payload):
        data = validate_search_payload(payload, top_k_max=self.config.top_k_max)
        if data['top_k'] == 0:
            return {'data': []}
        stage = 'query_budget'
        with self._request_lock, self.db._pair_lock:
            backend = self.db.pair_backend
            request_id = uuid.uuid4().hex
            try:
                parent_query = model_query(data['query'],
                    data['options'] if self.db.flags.get('use_options', True) else [], backend.tokenizer)
                source_query = compact(data['query'], backend.tokenizer)
                with request_context(request_id, parent_query) as state:
                    event('query_budget', 'bounded', parent_tokens=output.tokens(backend.tokenizer,parent_query),
                          source_tokens=output.tokens(backend.tokenizer,source_query),
                          shortened=source_query!=data['query'])
                    stage = 'recall'
                    result = self.db.search(user_id=data['user_id'], query=data['query'],
                                            top_k=100, options=data['options'])
                    if not result.results:
                        return {'data': []}
                    if any('fallback' in flag for r in result.results for flag in r.evidence_flags):
                        event('retrieval', 'retrieval_fallback', parents=len(result.results))
                    stage = 'source_validation'
                    with self.db.connection() as con:
                        resolved = output.resolve_sources(con, result.results,
                            user_id=data['user_id'], parent_limit=min(20, data['top_k']))
                    stage = 'source_candidates'
                    try:
                        candidates = output.make_candidates(resolved, source_query, backend.tokenizer,
                                                            role_tokens=ROLE_TOKENS)
                        stage = 'source_score'
                        scored = output.score_candidates(candidates, source_query, backend)
                    except Exception as error:
                        event(stage, 'raw_source_prefix_fallback', error,
                              sources=len(resolved['messages']))
                        # Fallback uses only validated original messages, never aggregate text.
                        raw = dict(resolved, parents=[dict(m,view='message',source_ids=[m['id']])
                                                     for m in resolved['messages']])
                        stage = 'source_prefix'
                        pack = output.prefix_output(raw, budget=8192, tokenizer=backend.tokenizer)
                    else:
                        stage = 'source_pack'
                        pack = output.selected_output(resolved, scored, budget=8192,
                                                      tokenizer=backend.tokenizer)
                    if not pack['pieces']:
                        return dict(data=[], **({'degraded':True} if state['degraded'] else {}))
                    digest = hashlib.sha256((data['user_id']+'\0'+pack['rendered']).encode()).hexdigest()
                    response = {'data': [{'id': 'source-pack-'+digest, 'content': pack['rendered']}]}
                    if state['degraded']:
                        response['degraded'] = True
                    return response
            except Exception as error:
                # Source integrity and storage failures must never become successful evidence.
                logger.error('fixed_window_failure request=%s stage=%s type=%s',
                             request_id, stage, type(error).__name__)
                raise ApiError(503, 'evidence service temporarily unavailable') from None

    def health(self):
        return dict(super().health(), evidence_profile=PROFILE)

def main():
    from .server import RetrieverServer
    config = RetrieverConfig.from_env().validate()
    service = FixedWindowService(config)
    httpd = RetrieverServer(config, service=service, quiet=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        service.close()

if __name__ == '__main__':
    main()
