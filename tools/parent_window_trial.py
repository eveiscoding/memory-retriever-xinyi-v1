"""Small paired short-aggregate index experiment on six reused public cases."""
import argparse,copy,hashlib,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import source_evidence_trial as core
from aml_retriever.config import RetrieverConfig
from aml_retriever.retriever import RetrieverDB
from aml_retriever import source_evidence_output as eo
from aml_retriever.parent_window_scoring import ParentWindowRetriever,windows
OLD=Path('/home/ubuntu/aml-source-evidence-expanded-20261030');SNAP=Path('/home/ubuntu/aml-embedding-funnel-effect-20261029/snapshots');ARMS=['baseline','parentmax']
PREVIOUS=Path('/home/ubuntu/aml-token-aggregate-effect-20261030-attempt2')

def settings(path):
 c=RetrieverConfig(db_path=str(path)).with_flags(views=True,vector=False,semantic_rerank=False,pair_rerank=True,source_aware_dedup=True)
 c.message_slot_ratio=.5;c.pair_rerank_mode='rrf';c.pair_rrf_weight=.25;c.pair_rrf_k=60.;c.pair_rerank_pool=100;return c.validate()
def original_hash(con,user):
 result={}
 for table,key in [('messages','id'),('views','view_id')]:
  extra=" AND view_type!='token-aggregate'" if table=='views' else ''
  rows=con.execute('SELECT * FROM '+table+' WHERE user_id=?'+extra+' ORDER BY '+key,(user,)).fetchall();result[table]=hashlib.sha256(json.dumps([list(r) for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 return result

def build_original(src,target,user):
 a=sqlite3.connect(src.as_uri()+'?mode=ro',uri=True);a.row_factory=sqlite3.Row;db=RetrieverDB(RetrieverConfig(db_path=str(target)).with_flags(vector=False));before=original_hash(a,user)
 def write(con):
  for table,key in [('messages','id'),('views','view_id')]:
   cols=[r[1] for r in con.execute('PRAGMA table_info('+table+')')];rows=a.execute('SELECT * FROM '+table+' WHERE user_id=? ORDER BY '+key,(user,)).fetchall()
   con.executemany('INSERT INTO '+table+'('+','.join(cols)+') VALUES('+','.join('?' for _ in cols)+')',[tuple(r[k] for k in cols) for r in rows])
   for r in rows:db._index_doc(con,r[key],user,'message' if table=='messages' else r['view_type'],r['content'])
  assert original_hash(con,user)==before
 db._write(write);a.close();db.close();return before

def clone(src,target):
 assert not target.exists();a=sqlite3.connect(src.as_uri()+'?mode=ro',uri=True);b=sqlite3.connect(str(target));a.backup(b);a.close();b.close()
class Dummy:
 def score(self,q,docs):raise AssertionError('preflight inference forbidden')
class Probe(RetrieverDB):
 def _pair_rerank(self,q,records):self.pool=copy.deepcopy(records);return records
 def _source_dedup_order(self,u,records):return records


def prepare(out):
 assert not core.git('status','--porcelain');old=json.loads((OLD/'prelock.json').read_text());prev=json.loads((PREVIOUS/'prelock.json').read_text());assert core.full()==old['full_before'];assert all(core.sha(p)==h for p,h in old['sources'].items());assert all(core.sha(p)==h for p,h in prev['sources'].items())
 base=core.model();assert base.model_hash==old['model_hash'];out.mkdir();(out/'work').mkdir();sources=dict(old['sources'])
 for path in [Path(__file__),ROOT/'aml_retriever/parent_window_scoring.py',ROOT/'configs/evidence-output/source-evidence-8k-v1.json',PREVIOUS/'input.json',PREVIOUS/'prelock.json',PREVIOUS/'summary.json']:sources[str(path)]=core.sha(path)
 cases=[];cache={};preflight_totals=[]
 for c in json.loads((PREVIOUS/'input.json').read_text()):
  src=PREVIOUS/'work'/(c['case_id']+'-baseline.db');sources[str(src)]=core.sha(src);pools={};plans={};pairmax=0
  for arm in ARMS:
   path=out/'work'/(c['case_id']+'-'+arm+'.db');clone(src,path);db=Probe(settings(path),pair_backend=Dummy());db.search(user_id=c['user_id'],query=c['query'],top_k=100);assert len(db.pool)==100
   pools[arm]=[dict(id=r['id'],content_sha256=hashlib.sha256(r['content'].encode()).hexdigest(),view=r['view']) for r in db.pool]
   assert pools[arm]==c['pools']['baseline']
   with db.connection() as con:assert original_hash(con,c['user_id'])==c['original_hash'];assert con.execute("SELECT COUNT(*) FROM views WHERE view_type='token-aggregate'").fetchone()[0]==0
   if arm=='parentmax':
    for r in db.pool:
     text=r['content'];ss=[None] if r['view']=='message' else windows(text,c['query'],base.tokenizer);plans[r['id']]=ss
     for span in ss:
      if span is not None:
       n=len(base.tokenizer(c['query'],text[span[0]:span[1]],add_special_tokens=True,truncation=False)['input_ids']);assert n<=512;pairmax=max(pairmax,n)
   db.close();sources[str(path)]=core.sha(path)
  assert pools['baseline']==pools['parentmax'];assert sum(map(len,plans.values()))<=5000
  cases.append({k:c[k] for k in ['case_id','user_id','query','gold','capability','subset','original_hash']}|dict(pools=pools,window_plans=plans,max_parent_pair=pairmax))
  for prior in [OLD,PREVIOUS]:
   f=prior/(c['case_id']+'-complete.json');sources[str(f)]=core.sha(f);x=json.loads(f.read_text());scored=x['scored_windows'] if prior==OLD else [z for rr in x['rows'] for z in rr['scored_output']]
   for z in scored:cache[hashlib.sha256((c['query']+'\0'+z['score_text']).encode()).hexdigest()]=z['score']
  count=sum(map(len,plans.values()));preflight_totals.append(count);print('PREFLIGHT',c['case_id'],'same_pool',len(pools['baseline']),'score_inputs',count,'max_pair',pairmax,flush=True)
 core.save(out/'input.json',cases);core.save(out/'score-cache.json',cache)
 for f in [out/'input.json',out/'score-cache.json']:sources[str(f)]=core.sha(f)
 lock=dict(code_sha=core.git('rev-parse','HEAD'),parent_sha='cf48406ff7b14eb3c633efe9e03e4dec1701c982',sources=sources,full_before=core.full(),db_metadata=old['db_metadata'],model_hash=base.model_hash,benchmark_sha=old['benchmark_sha'],budget=8192,aggregate_cap=384,overlap=64,max_new_pairs=10000,scope='Six reused public diagnostic cases. Identical fresh single-user original indexes, exact same pre-pair100 IDs/order/content. No child index. Only aggregate parent scoring uses internal bounded windows and max logit; raw messages retain original scores. Compare paired baseline only, not historical full-corpus numeric scores.',acceptance='Mean retrieval and8K metrics do not decline, no per-case gold-MRR20 decline or new Top5/20/visible20/8K gold losses. Zero fallback. Exact pre-pair membership/order/body and original data unchanged.',profile='Original RRF .25/k60, raw quota.5, same100 pair candidates; all aggregate windows scored max, cap384 overlap64 query-adaptive512. Winning span only for source visibility. Same source guard and8K source output. No embedding/topic/child recall. Extra model cost accepted; long-parent max bias not calibrated.',preflight_score_inputs=preflight_totals)
 core.save(out/'prelock.json',lock);print('PRELOCK',core.sha(out/'prelock.json'),flush=True)


class Cached:
 def __init__(self,base,bank,limit):self.base=base;self.tokenizer=base.tokenizer;self.bank=bank;self.new=0;self.reused=0;self.limit=limit;self.stage='search';self.costs=[]
 def score(self,q,texts):
  keys=[hashlib.sha256((q+'\0'+t).encode()).hexdigest() for t in texts];missing={k:t for k,t in zip(keys,texts) if k not in self.bank};reused=len(texts)-len(missing)
  if self.new+len(missing)>self.limit:raise ValueError('new pair budget exceeded')
  if missing:self.bank.update(zip(missing,self.base.score(q,list(missing.values()))));self.new+=len(missing)
  self.reused+=reused;self.costs.append(dict(stage=self.stage,new=len(missing),reused=reused));return [self.bank[k] for k in keys]
 def visible_spans(self,q,texts):return self.base.visible_spans(q,texts)
class Traced(RetrieverDB):
 def _pair_rerank(self,q,records):self.before_pair=copy.deepcopy(records);return super()._pair_rerank(q,records)
class ParentTraced(ParentWindowRetriever):
 def _pair_rerank(self,q,records):self.before_pair=copy.deepcopy(records);return super()._pair_rerank(q,records)

def run(out):
 lock=json.loads((out/'prelock.json').read_text());core.verify(lock);base=core.model();au=core.audit();assert base.model_hash==lock['model_hash'];sys.path.insert(0,'/home/ubuntu/aml-phase03.fLLf5PT8/benchmark');from benchmark import metrics
 pair=Cached(base,json.loads((out/'score-cache.json').read_text()),lock['max_new_pairs']);rows=[];started=time.monotonic()
 for c in json.loads((out/'input.json').read_text()):
  case_rows=[];begin=time.monotonic();before=pair.new
  for arm in ARMS:
   source=out/'work'/(c['case_id']+'-'+arm+'.db');target=out/'work'/(c['case_id']+'-'+arm+'-run.db');clone(source,target);pair.stage=arm+'-search';db=(Traced if arm=="baseline" else ParentTraced)(settings(target),pair_backend=pair);result=db.search(user_id=c['user_id'],query=c['query'],top_k=100)
   assert [r['id'] for r in db.before_pair]==[r['id'] for r in c['pools'][arm]];assert [hashlib.sha256(r['content'].encode()).hexdigest() for r in db.before_pair]==[r['content_sha256'] for r in c['pools'][arm]]
   assert not any('fallback' in f for r in result.results for f in r.evidence_flags);ids=[r.id for r in result.results];records=[dict(id=r.id,content=r.content) for r in result.results];docs={r.id:dict(content=r.content) for r in result.results};spans=base.visible_spans(c['query'],[r.content for r in result.results]) if arm=='baseline' else [next(t['best_span'] for t in db.parent_window_trace if t['id']==r.id) for r in result.results];visible={r.id:dict(content=r.content[a:b]) for r,(a,b) in zip(result.results,spans)}
   ranks=au.ranks(ids,docs,c['gold']);vranks=au.ranks(ids,visible,c['gold'])
   with db.connection() as con:
    assert original_hash(con,c['user_id'])==c['original_hash'];resolved=eo.resolve_sources(con,result.results,user_id=c['user_id'])
    for r in result.results:assert au.get_doc(con,r.id)['user_id']==c['user_id']
   candidates=eo.make_candidates(resolved,c['query'],base.tokenizer);pair.stage=arm+'-output';scored=eo.score_candidates(candidates,c['query'],pair);pack=eo.selected_output(resolved,scored,budget=8192,tokenizer=base.tokenizer);assert pack['used_tokens']<=8192;mm={m['id']:m for m in resolved['messages']}
   for s in pack['pieces']:assert s['user_id']==c['user_id'] and s['content']==mm[s['id']]['content'][s['start_char']:s['end_char']]
   hits=[any(au.covered(p['content'],g) for p in pack['pieces']) for g in c['gold']];pack.update(hits=hits,recall=sum(hits)/len(hits),complete=int(all(hits)))
   m=dict(mrr=metrics.reciprocal_rank(records,c['gold']),gold_mrr20=metrics.gold_reciprocal_rank_at_k(records,c['gold'],20),recall5=metrics.recall_at_k(records,c['gold'],5),recall20=metrics.recall_at_k(records,c['gold'],20),recall100=metrics.recall_at_k(records,c['gold'],100),complete20=metrics.complete_at_k(records,c['gold'],20),visible_recall20=sum(v is not None and v<=20 for v in vranks)/len(vranks))
   row=dict(case_id=c['case_id'],arm=arm,metrics=m,gold_ranks=ranks,visible_gold_ranks=vranks,pack=pack,ids=ids,child_top20=sum(r.view=='token-aggregate' for r in result.results[:20]),output_windows=len(scored),scored_output=scored,parent_window_trace=getattr(db,'parent_window_trace',[]));case_rows.append(row);rows.append(row);db.close()
  core.save(out/(c['case_id']+'-complete.json'),dict(rows=case_rows,new_pairs=pair.new-before,seconds=time.monotonic()-begin));print('CASE_COMPLETE',c['case_id'],pair.new-before,{r['arm']:[r['gold_ranks'],r['pack']['hits'],r['child_top20']] for r in case_rows},flush=True)
 n=len(rows)//2;keys=list(rows[0]['metrics']);scores={a:{k:sum(r['metrics'][k] for r in rows if r['arm']==a)/n for k in keys} for a in ARMS};packing={a:{k:sum(r['pack'][k] for r in rows if r['arm']==a)/n for k in ['recall','complete','used_tokens']} for a in ARMS};violations=[]
 for old in [r for r in rows if r['arm']=='baseline']:
  new=next(r for r in rows if r['arm']=='parentmax' and r['case_id']==old['case_id']);losses={str(k):[i for i,(a,b) in enumerate(zip(old['gold_ranks'],new['gold_ranks'])) if a is not None and a<=k and (b is None or b>k)] for k in [5,20]};vl=[i for i,(a,b) in enumerate(zip(old['visible_gold_ranks'],new['visible_gold_ranks'])) if a is not None and a<=20 and (b is None or b>20)];pl=[i for i,(a,b) in enumerate(zip(old['pack']['hits'],new['pack']['hits'])) if a and not b]
  if any(losses.values()) or vl or pl or new['metrics']['gold_mrr20']<old['metrics']['gold_mrr20']:violations.append(dict(case_id=old['case_id'],old_gold=old['gold_ranks'],new_gold=new['gold_ranks'],top_losses=losses,visible_loss=vl,pack_loss=pl))
 gate='PASS' if not violations and all(scores['parentmax'][k]>=scores['baseline'][k] for k in keys) and all(packing['parentmax'][k]>=packing['baseline'][k] for k in ['recall','complete']) else 'FAIL'
 summary=dict(status='complete',code_sha=lock['code_sha'],parent_sha=lock['parent_sha'],scores=scores,packing=packing,gate=gate,violations=violations,cost=dict(new_model_pairs=pair.new,reused_input_pairs=pair.reused,stages=pair.costs,api_calls=0,embedding_calls=0,fallbacks=0,seconds=time.monotonic()-started),scope=lock['scope']);core.save(out/'summary.json',summary);core.save(out/'scored-input-cache.json',pair.bank)
 lines=['# Parent-internal window scoring trial','','## Overall and detailed scores','',json.dumps({k:v for k,v in summary.items() if k!='cost'},indent=2),'','## Concrete cases','']
 for r in rows:lines.append(json.dumps(dict(case_id=r['case_id'],arm=r['arm'],ranks=r['gold_ranks'],visible=r['visible_gold_ranks'],pack_hits=r['pack']['hits'],child_top20=r['child_top20'])))
 (out/'report.md').open('x').write('\n'.join(lines)+'\n');core.verify(lock);core.save(out/'complete.json',dict(status='complete',gate=gate,code_sha=lock['code_sha'],full_unchanged=True,sources_unchanged=True,artifacts={p.name:core.sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='cost'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('out',type=Path);a=p.parse_args();prepare(a.out) if a.mode=='prepare' else run(a.out)
