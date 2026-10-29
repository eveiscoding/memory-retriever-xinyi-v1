"""Opt-in source evidence output. Search, ingestion and HTTP responses are unchanged.

Call resolve_sources under a read-only connection, then release it before model
scoring. No facts are generated. Every output piece is an exact original span.
"""
import copy,json,math


def _record(r):
 if isinstance(r,dict):return dict(r)
 return dict(id=r.id,user_id=r.user_id,view=r.view,content=r.content,created_at=r.created_at,source_ids=list(r.source_message_ids))


def resolve_sources(con,results,*,user_id,parent_limit=20):
 if not user_id or type(parent_limit) is not int or parent_limit<=0:raise ValueError('invalid scope')
 parents=[];messages={}
 for rank,item in enumerate(results[:parent_limit],1):
  r=_record(item)
  if r.get('user_id')!=user_id:raise ValueError('mixed tenant result')
  if r['view']=='message':
   row=con.execute('SELECT * FROM messages WHERE id=? AND user_id=?',(r['id'],user_id)).fetchone()
   if row is None or row['content']!=r['content']:raise ValueError('stale or missing message')
   ids=[r['id']]
  else:
   row=con.execute('SELECT * FROM views WHERE view_id=? AND user_id=?',(r['id'],user_id)).fetchone()
   if row is None or row['content']!=r['content']:raise ValueError('stale or missing view')
   ids=json.loads(row['source_ids'])
   supplied=r.get('source_ids',r.get('source_message_ids'))
   if supplied is not None and list(supplied)!=ids:raise ValueError('provenance mismatch')
  if not ids:raise ValueError('empty provenance')
  parent=dict(r,rank=rank,source_ids=ids);parents.append(parent)
  for mid in ids:
   m=con.execute('SELECT * FROM messages WHERE id=? AND user_id=?',(mid,user_id)).fetchone()
   if m is None or m['session_id']!=row['session_id']:raise ValueError('invalid source scope')
   m=dict(m)
   if mid not in messages:messages[mid]=dict(m,parent_ids=[],parent_rank=rank)
   if r['id'] not in messages[mid]['parent_ids']:messages[mid]['parent_ids'].append(r['id'])
 return dict(user_id=user_id,parents=parents,messages=list(messages.values()))


def tokens(tokenizer,text):return len(tokenizer.encode(text,add_special_tokens=False,truncation=False))

def _span(m,a,b,kind='anchor'):
 return dict(id=m['id'],user_id=m['user_id'],view='source-span',created_at=m['created_at'],roles=[m['role']],source_ids=[m['id']],parent_ids=list(m['parent_ids']),start_char=a,end_char=b,content=m['content'][a:b],kind=kind,session_id=m['session_id'],seq=m['seq'])


def make_candidates(resolved,query,tokenizer,*,max_pair_tokens=512,window_tokens=384,overlap=64,max_windows=5000):
 if not isinstance(query,str) or not query.strip():raise ValueError('query required')
 if not 0<=overlap<window_tokens or not 0<window_tokens<max_pair_tokens:raise ValueError('invalid window parameters')
 qlen=tokens(tokenizer,query);candidates=[]
 for m in resolved['messages']:
  role=(m['role'] or '')+': '
  room=min(window_tokens,max_pair_tokens-qlen-tokenizer.num_special_tokens_to_add(pair=True)-tokens(tokenizer,role)-8)
  if room<32:raise ValueError('query leaves insufficient pair room')
  text=m['content'];offsets=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)['offset_mapping']
  offsets=[(int(a),int(b)) for a,b in offsets if b>a]
  if not offsets:continue
  start=0;seen=set()
  while start<len(offsets):
   stop=min(len(offsets),start+room);a=0 if start==0 else offsets[start][0];b=len(text) if stop==len(offsets) else offsets[stop-1][1]
   score_text=role+text[a:b]
   while len(tokenizer(query,score_text,add_special_tokens=True,truncation=False)['input_ids'])>max_pair_tokens:
    stop-=1
    if stop<=start:raise ValueError('cannot construct bounded window')
    b=offsets[stop-1][1];score_text=role+text[a:b]
   if (a,b) not in seen:
    c=_span(m,a,b);c.update(parent_rank=m['parent_rank'],score_text=score_text,pair_tokens=len(tokenizer(query,score_text,add_special_tokens=True,truncation=False)['input_ids']))
    candidates.append(c);seen.add((a,b))
    if len(candidates)>max_windows:raise ValueError('source window limit exceeded')
   if stop==len(offsets):break
   nxt=max(start+1,stop-min(overlap,room-1))
   # Offset mappings may repeat for Unicode; advance until character progress.
   while nxt<len(offsets) and offsets[nxt][0]<=a:nxt+=1
   start=nxt
 return candidates


def score_candidates(candidates,query,scorer):
 unique={c['score_text']:None for c in candidates};texts=list(unique)
 values=scorer.score(query,texts) if texts else []
 if len(values)!=len(texts) or any(not math.isfinite(float(v)) for v in values):raise ValueError('invalid scorer output')
 bank=dict(zip(texts,map(float,values)))
 return [dict(c,score=bank[c['score_text']]) for c in candidates]


def render(pieces):
 blocks=[]
 for p in pieces:
  meta={k:p.get(k) for k in ['id','view','created_at','source_ids','roles','parent_ids','start_char','end_char']}
  blocks.append('EVIDENCE\n'+json.dumps(meta,ensure_ascii=False,separators=(',',':'))+'\nTEXT\n'+p['content']+'\nEND EVIDENCE\n')
 return '\n'.join(blocks)


def prefix_output(resolved,*,budget,tokenizer):
 if type(budget) is not int or budget<=0:raise ValueError('positive integer budget required')
 pieces=[]
 for r in resolved['parents']:
  p=dict(id=r['id'],view=r['view'],created_at=r.get('created_at'),source_ids=r['source_ids'],roles=sorted({m['role'] for m in resolved['messages'] if m['id'] in r['source_ids']}),parent_ids=[r['id']],start_char=0,end_char=len(r['content']),content=r['content'])
  if tokens(tokenizer,render(pieces+[p]))<=budget:pieces.append(p);continue
  ends=sorted({int(b) for a,b in tokenizer(p['content'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)['offset_mapping'] if b>a})
  lo=0;hi=len(ends);best=None
  while lo<hi:
   mid=(lo+hi)//2;q=dict(p,end_char=ends[mid],content=p['content'][:ends[mid]])
   if tokens(tokenizer,render(pieces+[q]))<=budget:best=q;lo=mid+1
   else:hi=mid
  if best is not None:pieces.append(best)
  break
 text=render(pieces);return dict(pieces=pieces,rendered=text,used_tokens=tokens(tokenizer,text),budget=budget)


def _merge_spans(spans,messages):
 grouped={};order=[]
 for s in spans:
  if s['id'] not in grouped:grouped[s['id']]=[];order.append(s['id'])
  grouped[s['id']].append(s)
 out=[]
 for mid in order:
  mm=messages[mid];intervals=sorted(grouped[mid],key=lambda s:(s['start_char'],s['end_char']));merged=[]
  for s in intervals:
   a,b=s['start_char'],s['end_char']
   if not 0<=a<b<=len(mm['content']) or s['content']!=mm['content'][a:b]:raise ValueError('non-original span')
   if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
   else:merged.append([a,b])
  out.extend(_span(mm,a,b,'selected') for a,b in merged)
 return out


def _neighbor_spans(c,messages,tokenizer,context_tokens):
 m=messages[c['id']];extra=[]
 # All neighbors must already belong to the selected parent result universe.
 # Short replies retain preceding context; assistant spans retain user question.
 short=tokens(tokenizer,m['content'])<=64
 if not(short or m['role']=='assistant') or c['start_char']>0:return extra
 prev=next((x for x in messages.values() if x['session_id']==m['session_id'] and x['seq']==m['seq']-1),None)
 if prev is not None:
  offsets=tokenizer(prev['content'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)['offset_mapping']
  a=offsets[max(0,len(offsets)-context_tokens)][0] if offsets else 0
  if a<len(prev['content']):extra.append(_span(prev,a,len(prev['content']),'context'))
 return extra


def selected_output(resolved,scored,*,budget,tokenizer,context_tokens=96):
 if type(budget) is not int or budget<=0 or type(context_tokens) is not int or context_tokens<1:raise ValueError('invalid output budget')
 messages={m['id']:m for m in resolved['messages']};chosen=[];trace=[]
 ordered=sorted(scored,key=lambda c:(-c['score'],c['parent_rank'],c['id'],c['start_char']))
 for c in ordered:
  if c['id'] not in messages or c.get('user_id')!=resolved['user_id']:raise ValueError('mixed tenant candidate')
  if not math.isfinite(c['score']):raise ValueError('non-finite score')
  candidate=_merge_spans(chosen+_neighbor_spans(c,messages,tokenizer,context_tokens)+[c],messages)
  if candidate==chosen:continue
  if tokens(tokenizer,render(candidate))<=budget:
   chosen=candidate;trace.append(dict(id=c['id'],start_char=c['start_char'],end_char=c['end_char'],score=c['score'],reason='selected'))
  else:trace.append(dict(id=c['id'],start_char=c['start_char'],end_char=c['end_char'],score=c['score'],reason='over_budget'))
 text=render(chosen);used=tokens(tokenizer,text)
 if used>budget:raise AssertionError('budget exceeded')
 return dict(pieces=chosen,rendered=text,used_tokens=used,budget=budget,trace=trace)
