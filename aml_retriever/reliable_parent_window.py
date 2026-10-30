"""HTTP-only bounded parent scorer; frozen offline experiment stays unchanged."""
import copy
from .parent_window_scoring import ParentWindowRetriever, score_records
from .retriever import RetrieverDB
from .pair_rerank import pair_order
from .model_input_budget import _context, compact, event

class ReliableParentWindowRetriever(ParentWindowRetriever):
    def _pair_rerank(self, query, records):
        if not records:
            return records
        with self._pair_lock:
            state = _context.get()
            bounded = state['parent_query'] if state else compact(query, self.pair_backend.tokenizer)
            try:
                count = min(len(records), self.config.pair_rerank_pool)
                scores, visible, trace = score_records(bounded, records[:count], self.pair_backend)
                cloned = copy.deepcopy(records)
                ordered = pair_order(cloned, scores, pool=count, mode=self.config.pair_rerank_mode,
                    weight=self.config.pair_rrf_weight, k=self.config.pair_rrf_k,
                    protect_top20=self.flags.get('pair_top20_protect',False),
                    protect_exact=self.flags.get('pair_exact_protect',False), query=query)
                for r in cloned[:count]:
                    r['_source_visible_span'] = visible[r['id']]
                    if r['view'] != 'message':
                        r['flags'].append('parent_window_scored')
                self.parent_window_trace = trace
                return ordered
            except Exception as error:
                event('parent_score', 'legacy_parent_fallback', error, parents=len(records))
                fallback = RetrieverDB._pair_rerank(self, bounded, copy.deepcopy(records))
                for r in fallback:
                    r['flags'].append('parent_window_fallback')
                return fallback
