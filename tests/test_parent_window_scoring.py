import copy,tempfile,unittest
from pathlib import Path
from tests.test_source_evidence_output import CharTokenizer
from aml_retriever.parent_window_scoring import windows,score_records,ParentWindowRetriever
from aml_retriever.retriever import RetrieverDB
from aml_retriever.config import RetrieverConfig

class Backend:
    identity='fixture';tokenizer=CharTokenizer()
    def score(self,q,texts):self.texts=list(texts);return [9. if 'TAIL' in t else 1. for t in texts]
    def visible_spans(self,q,texts):return [(0,min(8,len(t))) for t in texts]

def rec(i,text,view='session-segment'):
    return dict(id=i,view=view,content=text,user_id='u',source_ids=[i],score=1.,flags=[],channel_ranks={})

class ParentWindowTests(unittest.TestCase):
    def test_bounded_unicode_continuous_tail(self):
        text='中文😀 '+('x'*900)+' TAIL';tok=CharTokenizer();ss=windows(text,'q',tok,cap=64,overlap=8,max_pair=128)
        self.assertEqual(ss[0][0],0);self.assertEqual(ss[-1][1],len(text))
        for a,b in ss:self.assertLessEqual(len(tok('q',text[a:b])['input_ids']),128);self.assertLessEqual(len(text[a:b]),64)
        for a,b in zip(ss,ss[1:]):self.assertLessEqual(b[0],a[1])
    def test_query_adapts_room(self):
        tok=CharTokenizer();text='x'*900
        short=windows(text,'q',tok,cap=80,overlap=8,max_pair=128);long=windows(text,'q'*70,tok,cap=80,overlap=8,max_pair=128)
        self.assertGreater(short[0][1],long[0][1])
    def test_bad_profile_and_query(self):
        for q,cap,o in [('',64,8),('x'*200,64,8),('q',64,64)]:
            with self.assertRaises(ValueError):windows('abc',q,CharTokenizer(),cap=cap,overlap=o,max_pair=128)
    def test_parent_max_score_and_exact_winning_visibility(self):
        r=rec('p','x'*500+' TAIL');before=copy.deepcopy(r);b=Backend();scores,visible,trace=score_records('q',[r],b,cap=64,overlap=8,max_pair=128)
        self.assertEqual(scores,[9.]);a,z=visible['p'];self.assertIn('TAIL',r['content'][a:z]);self.assertEqual(trace[0]['best_span'],(a,z));self.assertEqual(r,before)
    def test_raw_message_still_one_original_input(self):
        r=rec('m','x'*600+' TAIL','message');b=Backend();s,v,t=score_records('q',[r],b)
        self.assertEqual(b.texts,[r['content']]);self.assertEqual(t[0]['windows'],1);self.assertEqual(v['m'],(0,8))
    def test_shared_model_inputs_keep_distinct_parents(self):
        b=Backend();rr=[rec('a','same'),rec('b','same')];s,v,t=score_records('q',rr,b)
        self.assertEqual(b.texts,['same']);self.assertEqual(len(s),2);self.assertEqual(set(v),{'a','b'})
    def test_tie_uses_first_span(self):
        b=Backend();r=rec('p','x'*800);s,v,t=score_records('q',[r],b);self.assertEqual(v['p'][0],0)
    def test_invalid_model_scores_and_budget_rejected(self):
        class Bad(Backend):
            def score(self,q,texts):return [float('nan')]*len(texts)
        with self.assertRaises(ValueError):score_records('q',[rec('p','abc')],Bad())
        with self.assertRaises(ValueError):score_records('q',[rec('p','x'*800)],Backend(),max_windows=1)
    def test_rerank_keeps_members_content_and_input_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=RetrieverConfig(db_path=str(Path(tmp)/'db')).with_flags(vector=False,pair_rerank=True,source_aware_dedup=True);db=ParentWindowRetriever(config,pair_backend=Backend())
            rr=[rec('p','x'*600+' TAIL'),rec('m','plain','message')];before=copy.deepcopy(rr);out=db._pair_rerank('q',rr)
            self.assertEqual(rr,before);self.assertEqual({r['id'] for r in out},{'p','m'});self.assertTrue(all('_source_visible_span' in r for r in out));self.assertFalse(any('fallback' in f for r in out for f in r['flags']));db.close()
    def test_failure_atomically_uses_legacy_parent_scores(self):
        class LegacyOnly(Backend):
            tokenizer=None
        with tempfile.TemporaryDirectory() as tmp:
            config=RetrieverConfig(db_path=str(Path(tmp)/'db')).with_flags(vector=False,pair_rerank=True,source_aware_dedup=True);db=ParentWindowRetriever(config,pair_backend=LegacyOnly());rr=[rec('p','TAIL'),rec('m','plain','message')];out=db._pair_rerank('q',rr)
            self.assertEqual({r['id'] for r in out},{'p','m'});self.assertTrue(all('parent_window_fallback' in r['flags'] for r in out));db.close()

    def test_actual_search_same_pool_and_no_child_index(self):
        class Capture(RetrieverDB):
            def _pair_rerank(self,q,records):self.pool=copy.deepcopy(records);return super()._pair_rerank(q,records)
        class CaptureParent(ParentWindowRetriever):
            def _pair_rerank(self,q,records):self.pool=copy.deepcopy(records);return super()._pair_rerank(q,records)
        with tempfile.TemporaryDirectory() as tmp:
            config=RetrieverConfig(db_path=str(Path(tmp)/'db')).with_flags(vector=False,pair_rerank=True,source_aware_dedup=True)
            a=Capture(config,pair_backend=Backend());a.add(request_id='a',user_id='u',session_id='s',messages=[dict(role='user',content='TAIL question'),dict(role='assistant',content='x'*900+' TAIL')]);a.add(request_id='b',user_id='v',session_id='s',messages=[dict(role='user',content='TAIL foreign')]);a.search(user_id='u',query='TAIL',top_k=100)
            b=CaptureParent(config,pair_backend=Backend());result=b.search(user_id='u',query='TAIL',top_k=100);self.assertEqual(a.pool,b.pool);self.assertEqual({r.id for r in result.results},{r['id'] for r in a.pool});self.assertTrue(all(r.user_id=='u' for r in result.results));self.assertFalse(any('fallback' in f for r in result.results for f in r.evidence_flags))
            with b.connection() as con:self.assertEqual(con.execute("SELECT COUNT(*) FROM views WHERE view_type='token-aggregate'").fetchone()[0],0)
            a.close();b.close()
