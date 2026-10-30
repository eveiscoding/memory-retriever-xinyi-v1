"""Explicit fixed-window + 8K source-evidence HTTP service.

The default CLI/service remains unchanged. Official data contains one item whose
content is the complete budgeted rendering used by the offline consumer.
"""
import hashlib
import threading
from .api import MemoryService, validate_search_payload
from .config import RetrieverConfig
from .parent_window_scoring import ParentWindowRetriever
from . import source_evidence_output as output

MODEL_HASH = '781f3f03fe04bf180ba0d912b6ee7f0177cf793176b5a317a8a5e0e8e7036559'
PROFILE = 'fixed-parent-384-64-max-source-8k-v1'

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
        with self._request_lock:
            result = self.db.search(user_id=data['user_id'], query=data['query'],
                                    top_k=100, options=data['options'])
            if not result.results:
                return {'data': []}
            if any('fallback' in flag for r in result.results for flag in r.evidence_flags):
                raise RuntimeError('fixed-window scoring unavailable')
            with self.db.connection() as con:
                resolved = output.resolve_sources(con, result.results,
                    user_id=data['user_id'], parent_limit=min(20, data['top_k']))
            # All tokenizer and graph use is serialized with the same owner lock.
            with self.db._pair_lock:
                backend = self.db.pair_backend
                candidates = output.make_candidates(resolved, data['query'], backend.tokenizer)
                scored = output.score_candidates(candidates, data['query'], backend)
                pack = output.selected_output(resolved, scored, budget=8192,
                                              tokenizer=backend.tokenizer)
            if not pack['pieces']:
                return {'data': []}
            digest = hashlib.sha256((data['user_id']+'\0'+pack['rendered']).encode()).hexdigest()
            return {'data': [{'id': 'source-pack-'+digest, 'content': pack['rendered']}]}

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
