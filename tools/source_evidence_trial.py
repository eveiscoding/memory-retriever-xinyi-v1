"""Six used public cases: fixed branch13 Search output, new source-only packing."""
import argparse,hashlib,importlib.util,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aml_retriever import source_evidence_output as eo
PRIOR=Path('/home/ubuntu/aml-message-slot-effect-20261029')
MODEL=Path('/home/ubuntu/phase03-mminilm-20261019/model')
AUDIT=Path('/home/ubuntu/evidence-diagnosis-20261028-v2.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.open('x').write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
def full():
 c=json.loads(subprocess.check_output(['docker','inspect','memory-api-rrf-indexed-parallel-888a3b9'],text=True))[0];assert c['State']['Running'] and c['State']['Health']['Status']=='healthy'
 return dict(id=c['Id'],image=c['Image'],mounts=sorted(c['Mounts'],key=lambda x:x['Destination']),started=c['State']['StartedAt'],restarts=c['RestartCount'],oom=c['State']['OOMKilled'])
def model():
 os.environ.update(USE_TORCH='0',USE_TF='0',TOKENIZERS_PARALLELISM='false',AML_PAIR_MODEL_PATH=str(MODEL),AML_PAIR_ONNX_FILE='onnx/model_quint8_avx2.onnx',AML_PAIR_MAX_LENGTH='512',AML_PAIR_BATCH_SIZE='1',AML_PAIR_THREADS='1')
 from aml_retriever.onnx_pair_backend import OnnxPairBackend
 b=OnnxPairBackend();b.tokenizer.model_max_length=10**9;return b
def audit():
 spec=importlib.util.spec_from_file_location('marker_audit',AUDIT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prelock(out):
 assert not git('status','--porcelain');assert not git('diff','060d9d044b8d90c26feadcd11af3b05fec184b4e','HEAD','--','aml_retriever/retriever.py','aml_retriever/config.py','aml_retriever/views.py')
 complete=json.loads((PRIOR/'complete.json').read_text());assert all(sha(PRIOR/n)==h for n,h in complete['artifacts'].items())
 old=json.loads((PRIOR/'prelock.json').read_text());assert all(sha(p)==h for p,h in old['sources'].items());assert full()==old['full_before']
 base=model();assert base.model_hash==old['model_hash'];out.mkdir();(out/'work').mkdir();cases=[];sources=dict(old['sources'])
 for p in [PRIOR/'complete.json',PRIOR/'prelock.json',PRIOR/'summary.json',AUDIT,Path(__file__),ROOT/'aml_retriever/source_evidence_output.py']:sources[str(p)]=sha(p)
 for c in json.loads((PRIOR/'trial-input.json').read_text()):
  prior=PRIOR/(c['case_id']+'-complete.json');sources[str(prior)]=sha(prior);r=next(r for r in json.loads(prior.read_text())['rows'] if r['arm']=='baseline')
  src=PRIOR/'indexes'/(c['subset']+'-mixed.db');target=out/'work'/(c['case_id']+'.db')
  a=sqlite3.connect(src.as_uri()+'?mode=ro',uri=True);b=sqlite3.connect(str(target));a.backup(b);b.close();a.close()
  con=sqlite3.connect(target.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;parents=[]
  for mid in r['ids'][:20]:
   m=con.execute('SELECT id,user_id,content,created_at FROM messages WHERE id=?',(mid,)).fetchone()
   if m is not None:p=dict(m,view='message',source_ids=[mid])
   else:
    m=con.execute('SELECT view_id AS id,user_id,view_type AS view,content,created_at,source_ids FROM views WHERE view_id=?',(mid,)).fetchone();p=dict(m);p['source_ids']=json.loads(p['source_ids'])
   assert p['user_id']==c['user_id'];parents.append(p)
  resolved=eo.resolve_sources(con,parents,user_id=c['user_id']);con.close();sources[str(target)]=sha(target)
  candidates=eo.make_candidates(resolved,c['query'],base.tokenizer)
  cases.append(dict(case_id=c['case_id'],query=c['query'],gold=c['gold'],capability=c['capability'],user_id=c['user_id'],resolved=resolved,candidates=candidates,parent_ids=r['ids'][:20],retrieval_metrics=r['metrics']))
  print('PREFLIGHT',c['case_id'],'sources',len(resolved['messages']),'windows',len(candidates),'max_pair',max(x['pair_tokens'] for x in candidates),flush=True)
 assert sum(len(c['candidates']) for c in cases)<=3000
 save(out/'input.json',cases);sources[str(out/'input.json')]=sha(out/'input.json')
 lock=dict(code_sha=git('rev-parse','HEAD'),parent_sha='060d9d044b8d90c26feadcd11af3b05fec184b4e',sources=sources,full_before=full(),db_metadata=old['db_metadata'],model_hash=base.model_hash,benchmark_sha=old['benchmark_sha'],budgets=[4096,8192],primary_budget=4096,scope='Six reused public development cases; freeze previous real branch13 Top20 Search IDs/order/body. Output-only experiment, not new retrieval or Answer/official score.',config=dict(parent_limit=20,window_tokens=384,overlap=64,max_pair=512,query_adaptive=True,context_tokens=96,short_reply_tokens=64,global_source_model_sort=True,per_parent_cap=None),acceptance='Primary4096: macro recall/complete no decline and at least one increase; no new per-case gold loss versus whole-view prefix. Report8192 separately. Zero generation/embedding/fallback; original spans and tenant isolation mandatory.',reuse='Frozen branch13 Search results reused, no Search inference; new local source-window scoring only.')
 save(out/'prelock.json',lock);print('PRELOCK',sha(out/'prelock.json'),flush=True)

def verify(lock):
 assert git('rev-parse','HEAD')==lock['code_sha'] and not git('status','--porcelain')
 assert all(sha(p)==h for p,h in lock['sources'].items())
 assert {p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in lock['db_metadata']}==lock['db_metadata'];assert full()==lock['full_before']

def run(out):
 lock=json.loads((out/'prelock.json').read_text());verify(lock);base=model();assert base.model_hash==lock['model_hash'];au=audit();start=time.monotonic();rows=[];pairs=0
 for c in json.loads((out/'input.json').read_text()):
  begin=time.monotonic();resolved=c['resolved'];original=json.dumps(resolved,sort_keys=True);scored=eo.score_candidates(c['candidates'],c['query'],base);calls=len({s['score_text'] for s in scored});pairs+=calls
  assert all(len(base.tokenizer(c['query'],s['score_text'],add_special_tokens=True,truncation=False)['input_ids'])<=512 for s in scored)
  packs={};messages={m['id']:m for m in resolved['messages']}
  for budget in lock['budgets']:
   packs[str(budget)]={}
   for arm in ['whole_view','source_output']:
    pack=eo.prefix_output(resolved,budget=budget,tokenizer=base.tokenizer) if arm=='whole_view' else eo.selected_output(resolved,scored,budget=budget,tokenizer=base.tokenizer)
    assert pack['used_tokens']<=budget
    if arm=='source_output':
     for s in pack['pieces']:
      m=messages[s['id']];assert s['user_id']==c['user_id'] and s['content']==m['content'][s['start_char']:s['end_char']]
    hits=[any(au.covered(s['content'],g) for s in pack['pieces']) for g in c['gold']]
    pack.update(hits=hits,recall=sum(hits)/len(hits),complete=int(all(hits)),chars=sum(len(s['content']) for s in pack['pieces']))
    packs[str(budget)][arm]=pack
  assert json.dumps(resolved,sort_keys=True)==original
  source_hits=[any(au.covered(m['content'],g) for m in resolved['messages']) for g in c['gold']]
  window_hits=[any(au.covered(s['content'],g) for s in scored) for g in c['gold']]
  row=dict(case_id=c['case_id'],packs=packs,source_hits=source_hits,window_hits=window_hits,source_count=len(messages),window_count=len(scored),unique_model_pairs=calls,seconds=time.monotonic()-begin,parent_ids=c['parent_ids'],retrieval_metrics=c['retrieval_metrics'],scored_windows=scored)
  save(out/(c['case_id']+'-complete.json'),row);rows.append(row);print('CASE_COMPLETE',c['case_id'],calls,row['seconds'],{b:{a:x['hits'] for a,x in v.items()} for b,v in packs.items()},flush=True)
 n=len(rows);scores={};violations={}
 for budget in map(str,lock['budgets']):
  scores[budget]={arm:{k:sum(r['packs'][budget][arm][k] for r in rows)/n for k in ['recall','complete','used_tokens','chars']} for arm in ['whole_view','source_output']}
  violations[budget]=[dict(case_id=r['case_id'],old=r['packs'][budget]['whole_view']['hits'],new=r['packs'][budget]['source_output']['hits']) for r in rows if any(a and not b for a,b in zip(r['packs'][budget]['whole_view']['hits'],r['packs'][budget]['source_output']['hits']))]
 primary=scores['4096'];gate='PASS' if not violations['4096'] and all(primary['source_output'][k]>=primary['whole_view'][k] for k in ['recall','complete']) and any(primary['source_output'][k]>primary['whole_view'][k] for k in ['recall','complete']) else 'FAIL'
 summary=dict(status='complete',code_sha=lock['code_sha'],scores=scores,violations=violations,gate=gate,unique_model_pairs=pairs,api_calls=0,embedding_calls=0,fallbacks=0,seconds=time.monotonic()-start,source_gold_groups=sum(sum(r['source_hits']) for r in rows),window_gold_groups=sum(sum(r['window_hits']) for r in rows),gold_groups=sum(len(r['source_hits']) for r in rows),scope=lock['scope'])
 save(out/'summary.json',summary)
 lines=['# Source evidence output trial','','## Overall and detailed scores','',json.dumps(summary,indent=2),'','## Concrete cases','']
 for r in rows:lines.append(json.dumps({k:v for k,v in r.items() if k not in ['packs','scored_windows']}));lines.append(json.dumps({b:{a:p['hits'] for a,p in v.items()} for b,v in r['packs'].items()}))
 (out/'report.md').open('x').write('\n'.join(lines)+'\n');verify(lock)
 save(out/'complete.json',dict(status='complete',code_sha=lock['code_sha'],gate=gate,full_unchanged=True,source_unchanged=True,artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prelock','run']);p.add_argument('out',type=Path);a=p.parse_args();prelock(a.out) if a.mode=='prelock' else run(a.out)
