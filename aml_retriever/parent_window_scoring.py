"""Explicit offline parent-window experiment; default RetrieverDB is unchanged."""
import copy, math
from .retriever import RetrieverDB
from .pair_rerank import pair_order
from .source_dedup import validate_visible_spans

def windows(text,query,tokenizer,*,cap=384,overlap=64,max_pair=512):
    if not query.strip() or not 0 <= overlap < cap < max_pair:
        raise ValueError('invalid query/window profile')
    room=min(cap,max_pair-len(tokenizer.encode(query,add_special_tokens=False,truncation=False))-tokenizer.num_special_tokens_to_add(pair=True)-8)
    if room < 32: raise ValueError('query leaves insufficient room')
    offsets=[(int(a),int(b)) for a,b in tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)['offset_mapping'] if b>a]
    if not offsets: return [(0,len(text))]
    spans=[];start=0
    while start<len(offsets):
        stop=min(len(offsets),start+room);a=0 if start==0 else offsets[start][0];b=len(text) if stop==len(offsets) else offsets[stop-1][1]
        while len(tokenizer(query,text[a:b],add_special_tokens=True,truncation=False)['input_ids'])>max_pair or len(tokenizer.encode(text[a:b],add_special_tokens=False,truncation=False))>cap:
            stop-=1
            if stop<=start: raise ValueError('cannot construct bounded original window')
            b=offsets[stop-1][1]
        if not spans or spans[-1]!=(a,b):spans.append((a,b))
        if stop==len(offsets):break
        nxt=max(start+1,stop-min(overlap,room-1))
        while nxt<len(offsets) and offsets[nxt][0]<=a:nxt+=1
        if nxt<len(offsets) and offsets[nxt][0]>b:raise ValueError('offset gap')
        start=nxt
    return spans

def score_records(query,records,backend,*,cap=384,overlap=64,max_pair=512,max_windows=5000):
    tokenizer=backend.tokenizer;plans=[];texts=[]
    for r in records:
        text=r['content'] or ''
        spans=[None] if r['view']=='message' else windows(text,query,tokenizer,cap=cap,overlap=overlap,max_pair=max_pair)
        plans.append(spans);texts.extend(text if s is None else text[s[0]:s[1]] for s in spans)
        if len(texts)>max_windows:raise ValueError('parent window limit exceeded')
    unique=list(dict.fromkeys(texts));values=list(backend.score(query,unique))
    if len(values)!=len(unique) or any(isinstance(v,bool) or not math.isfinite(float(v)) for v in values):raise ValueError('invalid window scores')
    bank=dict(zip(unique,map(float,values)));scores=[];visible={};trace=[]
    raw=[r for r in records if r['view']=='message']
    if raw:
        raw_visible=validate_visible_spans(raw,backend.visible_spans(query,[r['content'] or '' for r in raw]));visible.update(raw_visible)
    for r,spans in zip(records,plans):
        text=r['content'] or '';values=[bank[text if s is None else text[s[0]:s[1]]] for s in spans];best=max(range(len(values)),key=lambda i:values[i]);scores.append(values[best]);span=spans[best]
        if span is not None:visible[r['id']]=span
        trace.append(dict(id=r['id'],view=r['view'],windows=len(spans),best_span=visible[r['id']],score=values[best],window_spans=spans))
    return scores,visible,trace

class ParentWindowRetriever(RetrieverDB):
    """Opt-in Python experiment. No child index, Add, HTTP or default flag change."""
    def _pair_rerank(self,query,records):
        if not records:return records
        try:
            count=min(len(records),self.config.pair_rerank_pool)
            with self._pair_lock:
                scores,visible,trace=score_records(query,records[:count],self.pair_backend)
            cloned=copy.deepcopy(records)
            ordered=pair_order(cloned,scores,pool=count,mode=self.config.pair_rerank_mode,weight=self.config.pair_rrf_weight,k=self.config.pair_rrf_k,protect_top20=self.flags.get('pair_top20_protect',False),protect_exact=self.flags.get('pair_exact_protect',False),query=query)
            for r in cloned[:count]:
                r['_source_visible_span']=visible[r['id']]
                if r['view']!='message':r['flags'].append('parent_window_scored')
            self.parent_window_trace=trace
            return ordered
        except Exception:
            fallback=super()._pair_rerank(query,records)
            for r in fallback:r['flags'].append('parent_window_fallback')
            return fallback
