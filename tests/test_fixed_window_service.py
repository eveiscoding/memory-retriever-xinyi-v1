import tempfile,unittest,threading,json,urllib.request,urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from aml_retriever.config import RetrieverConfig
from aml_retriever.reliable_parent_window import ReliableParentWindowRetriever as ParentWindowRetriever
from aml_retriever.fixed_window_service import FixedWindowService
from aml_retriever import source_evidence_output as output
from aml_retriever.server import RetrieverServer
from tests.test_parent_window_scoring import Backend

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.cfg=RetrieverConfig(db_path=str(Path(self.tmp.name)/'db'),pair_rerank_mode='rrf',
            host='127.0.0.1',port=0,auth_mode='bearer',api_key='synthetic-test',retention_days=0
            ).with_flags(pair_rerank=True,source_aware_dedup=True,vector=False)
        self.db=ParentWindowRetriever(self.cfg,pair_backend=Backend())
        self.svc=FixedWindowService(self.cfg,self.db)
        self.payload=dict(request_id='r',user_id='u',session_id='s',messages=[
            dict(role='user',content='TAIL trip question'),
            dict(role='assistant',content='x'*900+' TAIL seven days.')])
        self.svc.official_add(self.payload)
    def tearDown(self):
        self.svc.close();self.tmp.cleanup()
    def query(self,user='u',k=100):return dict(user_id=user,query='TAIL trip',top_k=k)
    def test_transport_exactly_matches_offline_pack(self):
        results=self.db.search(user_id='u',query='TAIL trip',top_k=100).results
        with self.db.connection() as con:r=output.resolve_sources(con,results,user_id='u')
        b=self.db.pair_backend;c=output.make_candidates(r,'TAIL trip',b.tokenizer)
        pack=output.selected_output(r,output.score_candidates(c,'TAIL trip',b),budget=8192,tokenizer=b.tokenizer)
        actual=self.svc.official_search(self.query())['data']
        self.assertEqual(len(actual),1);self.assertEqual(actual[0]['content'],pack['rendered'])
        self.assertLessEqual(output.tokens(b.tokenizer,actual[0]['content']),8192)
        self.assertIn('TAIL seven days.',actual[0]['content'])
    def test_empty_zero_and_isolation(self):
        self.assertEqual(self.svc.official_search(self.query('other')),{'data':[]})
        self.assertEqual(self.svc.official_search(self.query(k=0)),{'data':[]})
        self.assertEqual(self.svc.official_add(self.payload)['success'],True)
        self.assertEqual(self.db.count('u'),2)
    def test_fallback_is_explicit(self):
        r=self.db.search(user_id='u',query='TAIL trip',top_k=100)
        r.results[0].evidence_flags.append('parent_window_fallback')
        with patch.object(self.db,'search',return_value=r):
            self.assertTrue(self.svc.official_search(self.query())['degraded'])
    def test_concurrent_queries_have_same_pack(self):
        expected=self.svc.official_search(self.query())
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows=list(ex.map(lambda _:self.svc.official_search(self.query()),range(8)))
        self.assertEqual(rows,[expected]*8)
    def test_profile_rejects_wrong_strategy(self):
        with self.assertRaises(ValueError):FixedWindowService(self.cfg.with_flags(pair_rerank=False),self.db)
    def test_http_auth_contract_and_no_admin(self):
        server=RetrieverServer(self.cfg,service=self.svc,quiet=True)
        th=threading.Thread(target=server.serve_forever,daemon=True);th.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            health=json.load(urllib.request.urlopen(base+'/health'))
            self.assertEqual(health['evidence_profile'],'fixed-parent-384-64-max-source-8k-query256-v2')
            req=urllib.request.Request(base+'/search',data=json.dumps(self.query()).encode(),headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as err:urllib.request.urlopen(req)
            self.assertEqual(err.exception.code,401)
            req.add_header('Authorization','Bearer synthetic-test')
            got=json.load(urllib.request.urlopen(req))
            self.assertEqual(got,self.svc.official_search(self.query()))
            with self.assertRaises(urllib.error.HTTPError) as err:urllib.request.urlopen(base+'/stats')
            self.assertEqual(err.exception.code,404)
        finally:server.shutdown();server.server_close();th.join()
