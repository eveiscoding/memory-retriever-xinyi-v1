import copy,json,math,sqlite3,tempfile,unittest
from pathlib import Path
from aml_retriever.config import RetrieverConfig
from aml_retriever.retriever import RetrieverDB
from aml_retriever import source_evidence_output as out

class CharTokenizer:
 def encode(self,text,**kw):return list(text)
 def num_special_tokens_to_add(self,pair=False):return 3 if pair else 2
 def __call__(self,a,b=None,**kw):
  if b is not None:return {'input_ids':[0]*(len(a)+len(b)+3)}
  return {'input_ids':list(a),'offset_mapping':[(i,i+1) for i in range(len(a))]}

class SourceEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.db=RetrieverDB(RetrieverConfig(db_path=str(Path(self.tmp.name)/'f.db')).with_flags(views=True,vector=False))
  self.db.add(request_id='fixture',user_id='u',session_id='s',messages=[dict(role='user',content='How long is the Japan trip?'),dict(role='assistant',content='Seven days.'),dict(role='user',content='And Chicago is three days.')])
  with self.db.connection() as c:
   r=c.execute('SELECT * FROM views ORDER BY view_id LIMIT 1').fetchone();self.result=dict(id=r['view_id'],view=r['view_type'],user_id='u',content=r['content'],created_at=r['created_at'],source_ids=json.loads(r['source_ids']))
   self.res=out.resolve_sources(c,[self.result],user_id='u')
  self.tok=CharTokenizer()
 def tearDown(self):self.db.close();self.tmp.cleanup()
 def candidates(self,res=None):return out.make_candidates(res or self.res,'trip',self.tok,max_pair_tokens=128,window_tokens=64,overlap=8)
 def test_scope_and_missing_sources(self):
  with self.db.connection() as c:
   with self.assertRaises(ValueError):out.resolve_sources(c,[self.result],user_id='other')
   r=dict(self.result,source_ids=['missing'])
   with self.assertRaises(ValueError):out.resolve_sources(c,[r],user_id='u')
 def test_stale_result_rejected(self):
  with self.db.connection() as c:
   with self.assertRaises(ValueError):out.resolve_sources(c,[dict(self.result,content='stale')],user_id='u')
 def test_unicode_tail_full_coverage_and_pair_bound(self):
  r=copy.deepcopy(self.res);m=r['messages'][0];m['content']='中文测试😀 '+('x'*300)+' TAIL';cc=self.candidates(r);cc=[x for x in cc if x['id']==m['id']]
  self.assertEqual(cc[0]['start_char'],0);self.assertEqual(cc[-1]['end_char'],len(m['content']))
  for x in cc:self.assertEqual(x['content'],m['content'][x['start_char']:x['end_char']]);self.assertLessEqual(x['pair_tokens'],128)
  for a,b in zip(cc,cc[1:]):self.assertLessEqual(b['start_char'],a['end_char'])
 def test_bad_scores_and_shared_text_do_not_merge_sources(self):
  class Bad:
   def score(self,q,d):return [float('nan')]*len(d)
  with self.assertRaises(ValueError):out.score_candidates(self.candidates(),'trip',Bad())
  class Good:
   def score(self,q,d):self.n=len(d);return [1.]*len(d)
  cs=self.candidates();cs=[cs[0],dict(cs[0],id='different')];g=Good();ss=out.score_candidates(cs,'trip',g)
  self.assertEqual(g.n,1);self.assertEqual(len(ss),2)
 def test_original_span_merge_and_gap_preservation(self):
  m=self.res['messages'][0];m['content']='abcdefghij';mm={m['id']:m}
  spans=[out._span(m,0,4),out._span(m,2,5),out._span(m,8,10)]
  merged=out._merge_spans(spans,mm);self.assertEqual([(x['start_char'],x['end_char']) for x in merged],[(0,5),(8,10)])
  with self.assertRaises(ValueError):out._merge_spans([dict(spans[0],content='fake')],mm)
 def test_short_reply_has_preceding_context(self):
  c=next(x for x in self.candidates() if x['roles']==['assistant']);mm={m['id']:m for m in self.res['messages']}
  n=out._neighbor_spans(c,mm,self.tok,96);self.assertEqual(len(n),1);self.assertEqual(n[0]['seq'],c['seq']-1)
 def test_exact_budget_and_no_mutation(self):
  original=copy.deepcopy(self.res);sc=[dict(c,score=float(i)) for i,c in enumerate(self.candidates())]
  for b in [1,500,2000]:
   p=out.selected_output(self.res,sc,budget=b,tokenizer=self.tok);self.assertLessEqual(p['used_tokens'],b);self.assertEqual(p['used_tokens'],len(p['rendered']))
  self.assertEqual(self.res,original)
 def test_foreign_candidate_rejected(self):
  c=dict(self.candidates()[0],score=1.,user_id='other')
  with self.assertRaises(ValueError):out.selected_output(self.res,[c],budget=1000,tokenizer=self.tok)
 def test_empty_and_overlong_query_rejected(self):
  for q in ['', 'q'*200]:
   with self.assertRaises(ValueError):out.make_candidates(self.res,q,self.tok,max_pair_tokens=128,window_tokens=64,overlap=8)
 def test_prefix_same_format_and_output_bound(self):
  for b in [1,500,2000]:
   p=out.prefix_output(self.res,budget=b,tokenizer=self.tok);self.assertLessEqual(p['used_tokens'],b)
   for x in p['pieces']:self.assertEqual(x['content'],self.result['content'][:x['end_char']])
  self.assertIn('\nTEXT\n',out.prefix_output(self.res,budget=2000,tokenizer=self.tok)['rendered'])
