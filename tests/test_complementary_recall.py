import copy
from datetime import datetime, timezone
import math
import os
import tempfile
import unittest
from unittest.mock import patch

from aml_retriever.api import MemoryService
from aml_retriever.config import RetrieverConfig
from aml_retriever.retriever import RetrieverDB
from tests.test_temporal_facts import EchoFacts, add_fact


def stamp(text):
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


class SemanticFixture:
    identity = model_hash = "stub-not-a-quality-model"
    dimension = 2
    def __init__(self):
        self.calls = []
        self.bad = None
    def embed(self, texts):
        self.calls.append(list(texts))
        if self.bad == "raise":
            raise RuntimeError("must not leak")
        if self.bad == "shape":
            return [[1]] * len(texts)
        if self.bad == "nan":
            return [[math.nan, 0]] * len(texts)
        if self.bad == "huge":
            return [[1e308, 1e308]] * len(texts)
        return [[1., 0.] if "physician" in t or "doctor" in t else [0., 1.] for t in texts]


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RetrieverConfig().with_flags(fact_extraction=True, fact_versions=True,
                                               complementary_recall=True, views=False)
        self.db = RetrieverDB(self.cfg, fact_backend=EchoFacts())
        self.alice = add_fact(self.db, "Alice joins Orion.")
        self.orion = add_fact(self.db, "Orion launches a telescope.")
        self.doctor = add_fact(self.db, "A doctor treats patients.")
        self.link(self.alice, "Alice")
        self.link(self.alice, "Orion")
        self.link(self.orion, "Orion")

    def tearDown(self):
        self.db.close()

    def link(self, fid, entity):
        f = next(f for f in self.db.list_facts(user_id="a") if f["fact_id"] == fid)
        s = f["sources"][0]
        self.db.link_fact_entity(user_id="a", fact_id=fid, entity=entity,
                                 message_id=s["message_id"], quote=s["quote"])

    def recall(self, query="Alice", **kwargs):
        return self.db.recall_facts(user_id="a", query=query, **kwargs)

    def test_independent_keyword_and_graph_complement_and_rrf(self):
        result = self.recall()
        self.assertEqual(result["channel_orders"]["keyword"], [self.alice])
        self.assertEqual(result["channel_orders"]["graph"], [self.alice, self.orion])
        self.assertEqual(result["results"][0]["fact_id"], self.alice)
        self.assertAlmostEqual(result["results"][0]["score"], 2 / 61)
        self.assertAlmostEqual(result["results"][1]["score"], 1 / 62)
        self.assertEqual(result["channels"]["semantic"], "unavailable")
        self.assertEqual(result["channels"]["temporal"], "not_requested")

    def test_each_channel_can_be_independently_disabled(self):
        for name in ("keyword", "graph", "temporal", "semantic"):
            self.db.flags["fact_recall_" + name] = False
        result = self.recall()
        self.assertEqual(result["results"], [])
        self.assertEqual(set(result["channels"].values()), {"disabled"})
        self.db.flags["fact_recall_graph"] = True
        self.assertEqual(len(self.recall()["results"]), 2)
        self.db.flags["fact_recall_graph"] = False
        self.db.flags["fact_recall_keyword"] = True
        self.assertEqual(len(self.recall()["results"]), 1)

    def test_graph_hop_bound_and_no_substring_entity_seeds(self):
        self.cfg.fact_recall_graph_hops = 1
        self.assertEqual(self.recall()["channel_orders"]["graph"], [self.alice])
        self.assertEqual(self.recall("AliceXYZ")["channel_orders"]["graph"], [])

    def test_chinese_explicit_entity_labels_work_without_spaces(self):
        first = add_fact(self.db, "小明加入星河。")
        second = add_fact(self.db, "星河位于上海。")
        self.link(first, "小明")
        self.link(first, "星河")
        self.link(second, "星河")
        self.assertEqual(set(self.recall("小明在哪工作")["channel_orders"]["graph"]), {first,second})

    def test_graph_entity_citations_and_user_scope(self):
        raw = self.db.list_facts(user_id="a")[0]["sources"][0]
        s = {key:raw[key] for key in ("message_id", "quote")}
        for args in [dict(user_id="b", fact_id=self.alice, entity="Alice", **s),
                     dict(user_id="a", fact_id=self.alice, entity="Fabricated", **s)]:
            with self.assertRaises(ValueError):
                self.db.link_fact_entity(**args)
        bob = add_fact(self.db, "Alice private Bob secret", user="b")
        self.assertNotIn(bob, [r["fact_id"] for r in self.recall()["results"]])
        self.assertEqual(self.db.recall_facts(user_id="nobody", query="Alice")["results"], [])

    def test_semantic_stub_complements_no_keyword_match(self):
        backend = SemanticFixture()
        self.db.embedding_backend = backend
        result = self.recall("physician")
        self.assertEqual(result["channel_orders"]["keyword"], [])
        self.assertEqual(result["channel_orders"]["semantic"], [self.doctor])
        self.assertEqual(result["results"][0]["fact_id"], self.doctor)
        self.assertEqual(len(backend.calls), 1)
        self.assertFalse(self.db.embedding_enabled)

    def test_semantic_errors_are_atomic_and_other_channels_survive(self):
        backend = SemanticFixture()
        self.db.embedding_backend = backend
        for mode in ("raise", "shape", "nan", "huge"):
            backend.bad = mode
            result = self.recall()
            self.assertEqual(result["channels"]["semantic"], "failed")
            self.assertEqual(result["channel_orders"]["semantic"], [])
            self.assertEqual(result["results"][0]["fact_id"], self.alice)
            self.assertNotIn("must not leak", str(result))

    def test_semantic_budget_does_not_call_backend(self):
        backend = SemanticFixture()
        self.db.embedding_backend = backend
        self.cfg.fact_recall_embed_chars = 1
        self.assertEqual(self.recall()["channels"]["semantic"], "input_limit")
        self.assertEqual(backend.calls, [])

    def test_semantic_only_sees_this_user(self):
        backend = SemanticFixture()
        self.db.embedding_backend = backend
        add_fact(self.db, "ConfidentialBob", user="b")
        self.recall()
        self.assertFalse(any("ConfidentialBob" in t for t in backend.calls[0]))

    def test_temporal_point_versions_and_unknown_excluded(self):
        first = self.db.record_fact_version(user_id="a", fact_id=self.alice, operation_id="v1",
            version_key="Alice:employment", valid_at=stamp("2020-01-01"))
        self.db.record_fact_version(user_id="a", fact_id=self.orion, operation_id="v2",
            version_key="Alice:employment", valid_at=stamp("2021-01-01"), supersedes=first["version_id"])
        old = self.recall("On 2020-06-01")
        new = self.recall("On 2021-01-01")
        self.assertEqual(old["channel_orders"]["temporal"], [self.alice])
        self.assertEqual(new["channel_orders"]["temporal"], [self.orion])
        self.assertNotIn(self.doctor, [r["fact_id"] for r in new["results"]])

    def test_explicit_time_point_without_date_text(self):
        self.db.record_fact_version(user_id="a", fact_id=self.alice, operation_id="v1",
            version_key="key", valid_at=100, invalid_at=200)
        result = self.recall("Alice", valid_at=150)
        self.assertEqual(result["valid_at"], 150)
        self.assertEqual([r["fact_id"] for r in result["results"]], [self.alice])

    def test_temporal_requires_versions_and_dates_not_guessed(self):
        for text in ("On 2020-99-99", "2020-01-01 through 2021-01-01"):
            with self.assertRaises(ValueError):
                self.recall(text)
        self.assertIsNone(self.recall("yesterday")["valid_at"])
        self.db.flags["fact_versions"] = False
        with self.assertRaises(ValueError):
            self.recall("2020-01-01")

    def test_scan_result_and_channel_budgets_are_observable(self):
        self.cfg.fact_recall_scan_limit = 1
        result = self.recall()
        self.assertTrue(result["truncated"])
        self.assertEqual(result["eligible_facts"], 1)
        self.cfg.fact_recall_scan_limit = 100
        self.cfg.fact_recall_channel_limit = 1
        result = self.recall(limit=1)
        self.assertEqual(len(result["results"]), 1)
        self.assertTrue(all(len(v) <= 1 for v in result["channel_orders"].values()))

    def test_search_projects_original_message_not_derived_text(self):
        self.db.flags["complementary_recall"] = False
        before = self.db.search(user_id="a", query="Alice", top_k=100)
        self.db.flags["complementary_recall"] = True
        after = self.db.search(user_id="a", query="Alice", top_k=100)
        self.assertFalse(any("telescope" in r.content for r in before.results))
        expanded = next(r for r in after.results if "telescope" in r.content)
        self.assertEqual(expanded.view, "message")
        self.assertEqual(expanded.source_message_ids, [expanded.id])
        self.assertIn("fact_graph", expanded.channel_ranks)
        self.assertIn("fact_fused", expanded.channel_ranks)
        self.assertEqual(expanded.recall_status["semantic"], "unavailable")

    def test_no_fact_data_leaves_baseline_unchanged_and_query_failure_falls_back(self):
        self.db.flags["complementary_recall"] = False
        old = self.db.search(user_id="a", query="Alice", top_k=100)
        self.db.flags["complementary_recall"] = True
        with patch("aml_retriever.complementary_recall.recall", side_effect=RuntimeError("hidden")):
            result = self.db.search(user_id="a", query="Alice", top_k=100)
        self.assertEqual([(r.id,r.score) for r in old.results], [(r.id,r.score) for r in result.results])
        self.assertEqual(result.results[0].recall_status, {"complementary":"failed"})

    def test_http_contract_projection_and_provenance_toggle_without_network(self):
        service = MemoryService(self.cfg, db=self.db)
        service.include_provenance = True
        response = service.official_search(dict(user_id="a", query="Alice", top_k=100))
        row = next(r for r in response["data"] if "telescope" in r["content"])
        self.assertEqual(row["view"], "message")
        self.assertIn("recall_status", row)
        self.assertNotIn("derived", row)
        service.include_provenance = False
        response = service.official_search(dict(user_id="a", query="Alice", top_k=100))
        self.assertTrue(all("recall_status" not in r and "channel_ranks" not in r for r in response["data"]))

    def test_no_facts_available_baseline_and_non_temporal_empty(self):
        # Raw-only histories do not magically become facts on feature enable.
        self.db.facts_enabled = False
        self.db.add(user_id="raw", request_id="r", session_id="s", messages=[dict(role="user",content="Evidence")])
        self.db.flags["complementary_recall"] = False
        before = self.db.search(user_id="raw", query="Evidence", top_k=100)
        self.db.flags["complementary_recall"] = True
        after = self.db.search(user_id="raw", query="Evidence", top_k=100)
        self.assertEqual([(r.id,r.score) for r in before.results], [(r.id,r.score) for r in after.results])
        self.assertEqual(self.db.recall_facts(user_id="raw", query="Evidence")["results"], [])

    def test_knowledge_cutoff_excludes_future_source_and_entity_links(self):
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.db._write(lambda con: con.execute("UPDATE messages SET added_at='2999-01-01' WHERE user_id='a'"))
        self.assertEqual(self.recall(known_at=now)["results"], [])
        self.db._write(lambda con: con.execute("UPDATE messages SET added_at='2000-01-01' WHERE user_id='a'"))
        self.db._write(lambda con: con.execute("UPDATE fact_entities SET recorded_at=?", (now+10000,)))
        result = self.recall(known_at=now)
        self.assertEqual(result["channel_orders"]["graph"], [])
        self.assertEqual(result["channel_orders"]["keyword"], [self.alice])

    def test_future_version_annotation_does_not_rewrite_unversioned_past(self):
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with patch("aml_retriever.temporal_facts.now_ms", return_value=now+10000):
            self.db.record_fact_version(user_id="a", fact_id=self.alice, operation_id="future-note",
                version_key="employment", valid_at=stamp("2999-01-01"))
        past = self.recall(known_at=now)
        self.assertIn(self.alice, [r["fact_id"] for r in past["results"]])

    def test_master_off_is_default_and_internal_operations_guarded(self):
        db = RetrieverDB()
        try:
            self.assertFalse(db.flags["fact_versions"])
            self.assertFalse(db.flags["complementary_recall"])
            with self.assertRaises(ValueError):
                db.recall_facts(user_id="a", query="Alice")
            with self.assertRaises(ValueError):
                db.link_fact_entity(user_id="a", fact_id="f", entity="Alice", message_id="m", quote="Alice")
        finally:
            db.close()

    def test_derived_indices_and_links_removed_on_delete_ttl_and_purge(self):
        self.db.delete_user("a")
        for table in ("fact_fts", "fact_entities"):
            self.assertEqual(len(self.db.query(f"SELECT * FROM {table}")), 0)
        add_fact(self.db, "Stale")
        self.db._write(lambda con: con.execute("UPDATE messages SET added_at='2000-01-01'"))
        self.db.purge_stale_users(1)
        self.assertEqual(len(self.db.query("SELECT * FROM fact_fts")), 0)
        add_fact(self.db, "Final")
        self.db.purge_all()
        self.assertEqual(len(self.db.query("SELECT * FROM fact_fts")), 0)

    def test_invalid_budgets(self):
        for field, value in [("fact_recall_scan_limit",0), ("fact_recall_channel_limit",401),
                ("fact_recall_graph_hops",3), ("fact_recall_embed_chars",0),
                ("fact_recall_rrf_k",True), ("fact_recall_weight",math.nan)]:
            cfg = RetrieverConfig()
            setattr(cfg, field, value)
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_old_fact_index_backfilled_on_explicit_enable(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "test.db")
            db = RetrieverDB(self.cfg, db_path=path, fact_backend=EchoFacts())
            add_fact(db, "Persistent")
            db._write(lambda con: con.execute("DELETE FROM fact_fts"))
            db.close()
            db = RetrieverDB(self.cfg, db_path=path, fact_backend=EchoFacts())
            try:
                self.assertEqual(len(db.recall_facts(user_id="a", query="Persistent")["results"]), 1)
                self.assertEqual(len(db.query("SELECT * FROM fact_fts")), 1)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
