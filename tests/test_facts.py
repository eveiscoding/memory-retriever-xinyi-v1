import copy
import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from aml_retriever.config import RetrieverConfig
from aml_retriever.facts import fact_id, load_fact_backend, validate_facts
from aml_retriever.retriever import IdempotencyConflictError, RetrieverDB


MESSAGES = [dict(role="user", content="I prefer tea.", timestamp=1700000000000)]


def payload(text="The user prefers tea.", index=0, quote="I prefer tea."):
    return {"facts": [dict(text=text, kind="preference",
                           sources=[dict(message_index=index, quote=quote)])]}


class Backend:
    identity = "fixture-only-not-a-model"

    def __init__(self, value=None):
        self.value = value if value is not None else payload()
        self.calls = []

    def extract(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if isinstance(self.value, Exception):
            raise self.value
        return copy.deepcopy(self.value)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.messages = [dict(role="user", content="I prefer tea.", ts_ms=1)]

    def test_schema_and_source_validation(self):
        invalid = [[], {"facts": "bad"}, {"facts": [], "user_id": "other"},
                   payload(index=True), payload(index=-1), payload(index=1),
                   payload(quote="not in the original"), payload(quote=""),
                   payload(text=""), payload(text="a" * 1025),
                   {"facts": [dict(text="x", kind="preference", sources=[])]}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                validate_facts(value, self.messages, 32)

    def test_json_strict_not_markdown_or_over_budget(self):
        self.assertEqual(len(validate_facts(json.dumps(payload()), self.messages, 1)), 1)
        for value in ["```json\n{}\n```", "x" * 65537, {"facts": payload()["facts"] * 2}]:
            with self.assertRaises(ValueError):
                validate_facts(value, self.messages, 1)

    def test_role_boundary(self):
        messages = self.messages + [dict(role="assistant", content="Try coffee.", ts_ms=2)]
        value = payload()
        value["facts"][0]["sources"].append(dict(message_index=1, quote="Try coffee."))
        with self.assertRaises(ValueError):
            validate_facts(value, messages, 32)
        messages[0]["role"] = "system"
        with self.assertRaises(ValueError):
            validate_facts(payload(), messages, 32)

    def test_ids_are_scoped_and_dedup_is_not_semantic(self):
        self.assertEqual(fact_id("a", "user", "I prefer tea."),
                         fact_id("a", "user", " I   prefer tea. "))
        ids = {fact_id(u, r, t) for u, r, t in [
            ("a", "user", "I prefer tea."), ("b", "user", "I prefer tea."),
            ("a", "assistant", "I prefer tea."), ("a", "user", "I dislike tea.")]}
        self.assertEqual(len(ids), 4)
        self.assertNotEqual(fact_id("a", "user", "US"), fact_id("a", "user", "us"))

    def test_config(self):
        with patch.dict(os.environ, {"AML_FLAG_FACT_EXTRACTION": "1",
                "AML_FACT_BACKEND": "sample:factory", "AML_FACT_MAX_FACTS": "8"}):
            cfg = RetrieverConfig.from_env().validate()
        self.assertTrue(cfg.flags["fact_extraction"])
        self.assertEqual((cfg.fact_backend, cfg.fact_max_facts), ("sample:factory", 8))
        for field in ["fact_max_facts", "fact_max_input_chars"]:
            cfg = RetrieverConfig()
            setattr(cfg, field, 0)
            with self.assertRaises(ValueError):
                cfg.validate()
        with self.assertRaises(ValueError):
            load_fact_backend("missing-colon")
        with self.assertRaises(ValueError):
            RetrieverDB(RetrieverConfig().with_flags(fact_extraction=True))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.db = RetrieverDB(RetrieverConfig().with_flags(fact_extraction=True), fact_backend=self.backend)

    def tearDown(self):
        self.db.close()

    def add(self, request="r", user="alice", messages=None):
        return self.db.add(request_id=request, user_id=user, session_id="s",
                           messages=copy.deepcopy(MESSAGES if messages is None else messages))

    def test_atomic_add_provenance_and_shadow_only(self):
        result = self.add()
        self.assertEqual(result.fact_status, "extracted")
        facts = self.db.list_facts(user_id="alice")
        self.assertEqual([f["fact_id"] for f in facts], result.fact_ids)
        self.assertEqual(facts[0]["source_role"], "user")
        self.assertEqual(facts[0]["sources"][0]["message_id"], result.message_ids[0])
        self.assertEqual(facts[0]["sources"][0]["quote"], MESSAGES[0]["content"])
        self.assertEqual(facts[0]["sources"][0]["ts_ms"], MESSAGES[0]["timestamp"])
        self.assertTrue(facts[0]["derived"])
        self.assertEqual(self.db.query("SELECT COUNT(*) FROM fts WHERE doc_type='fact'")[0][0], 0)
        search = self.db.search(user_id="alice", query="tea", top_k=100)
        self.assertTrue(any(r.id == result.message_ids[0] for r in search.results))
        self.assertTrue(all(r.view != "fact" for r in search.results))
        self.assertEqual(self.backend.calls[0]["messages"][0]["message_index"], 0)

    def test_idempotent_no_reextract_and_conflict(self):
        original = self.add()
        retry = self.add()
        self.assertTrue(retry.idempotent)
        self.assertEqual(retry.fact_ids, original.fact_ids)
        self.assertEqual(len(self.backend.calls), 1)
        with self.assertRaises(IdempotencyConflictError):
            self.add(messages=[dict(role="user", content="New content")])
        self.assertEqual(len(self.backend.calls), 1)

    def test_duplicate_facts_keep_all_sources_and_originals(self):
        self.backend.value["facts"] *= 2
        first, second = self.add(), self.add("r2")
        self.assertEqual(first.fact_ids, second.fact_ids)
        self.assertEqual(len(first.fact_ids), 1)
        facts = self.db.list_facts(user_id="alice")
        self.assertEqual(len(facts), 1)
        self.assertEqual(len(facts[0]["sources"]), 2)
        self.assertEqual(self.db.count("alice"), 2)

    def test_speaker_not_conflated(self):
        self.add()
        self.add("r2", messages=[dict(role="assistant", content="I prefer tea.")])
        self.assertEqual({f["source_role"] for f in self.db.list_facts(user_id="alice")},
                         {"user", "assistant"})

    def test_changed_fact_does_not_delete_history(self):
        self.add()
        self.backend.value = payload(text="The user no longer prefers tea.", quote="I no longer prefer tea.")
        self.add("r2", messages=[dict(role="user", content="I no longer prefer tea.")])
        self.assertEqual(len(self.db.list_facts(user_id="alice")), 2)
        self.assertEqual(self.db.count("alice"), 2)

    def test_invalid_output_fails_whole_batch_but_preserves_raw(self):
        self.backend.value["facts"].append(payload(quote="fabricated")["facts"][0])
        result = self.add()
        self.assertEqual(result.fact_status, "failed")
        self.assertEqual(result.fact_ids, [])
        self.assertEqual(self.db.list_facts(user_id="alice"), [])
        self.assertEqual(self.db.count("alice"), 1)
        self.assertEqual(self.db.query("SELECT error_code FROM fact_extractions")[0][0], "invalid_output")

    def test_backend_error_redacted_and_retry_stable(self):
        self.backend.value = RuntimeError("secret and source text must not leak")
        result = self.add()
        self.assertEqual(result.fact_status, "failed")
        row = dict(self.db.query("SELECT * FROM fact_extractions")[0])
        self.assertNotIn("secret", str(row))
        self.assertEqual(row["error_code"], "backend_error")
        self.assertTrue(self.add().idempotent)
        self.assertEqual(len(self.backend.calls), 1)

    def test_input_budget_and_no_user_or_assistant_skip_backend(self):
        self.db.config.fact_max_input_chars = 2
        self.assertEqual(self.add().fact_status, "input_limit")
        self.assertEqual(self.add("r2", messages=[dict(role="system", content="hidden")]).fact_status, "no_facts")
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.db.count("alice"), 2)

    def test_system_filtered_but_original_indices_kept(self):
        self.backend.value = payload(index=1)
        result = self.add(messages=[dict(role="system", content="private instruction")] + MESSAGES)
        self.assertEqual(result.fact_status, "extracted")
        inputs = self.backend.calls[0]["messages"]
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0]["message_index"], 1)

    def test_user_isolation_and_delete(self):
        self.add()
        self.add(user="bob")
        self.assertEqual(self.db.list_facts(user_id="nobody"), [])
        self.db.delete_user("alice")
        for table in ("facts", "fact_sources", "fact_extractions"):
            self.assertEqual(self.db.query(f"SELECT COUNT(*) FROM {table} WHERE user_id='alice'")[0][0], 0)
            self.assertTrue(self.db.query(f"SELECT COUNT(*) FROM {table} WHERE user_id='bob'")[0][0])
        self.assertEqual(len(self.db.list_facts(user_id="bob")), 1)
        with self.assertRaises(ValueError):
            self.db.list_facts(user_id="")
        with self.assertRaises(ValueError):
            self.db.list_facts(user_id="bob", limit=True)

    def test_retention_and_purge_remove_derivatives(self):
        self.add()
        self.add(user="bob")
        self.db._write(lambda con: con.execute(
            "UPDATE messages SET added_at='2000-01-01' WHERE user_id='alice'"))
        self.assertEqual(self.db.purge_stale_users(1)["deleted_users"], 1)
        self.assertEqual(self.db.list_facts(user_id="alice"), [])
        self.assertEqual(len(self.db.list_facts(user_id="bob")), 1)
        self.db.purge_all()
        for table in ("facts", "fact_sources", "fact_extractions"):
            self.assertEqual(self.db.query(f"SELECT COUNT(*) FROM {table}")[0][0], 0)

    def test_concurrent_retry_calls_backend_once(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.add(), range(16)))
        self.assertEqual(sum(not r.idempotent for r in results), 1)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.db.count(), 1)

    def test_storage_failure_rolls_back_raw_facts_and_request(self):
        self.db._write(lambda con: con.execute("""CREATE TRIGGER reject_fact BEFORE INSERT ON facts
            BEGIN SELECT RAISE(ABORT, 'fixture failure'); END"""))
        with self.assertRaises(sqlite3.IntegrityError):
            self.add()
        for table in ("facts", "messages", "requests", "fact_extractions", "fact_sources", "fts"):
            self.assertEqual(self.db.query(f"SELECT COUNT(*) FROM {table}")[0][0], 0)


class DisabledAndMigrationTests(unittest.TestCase):
    def test_shadow_extraction_does_not_change_rankings(self):
        baseline = RetrieverDB()
        shadow = RetrieverDB(RetrieverConfig().with_flags(fact_extraction=True), fact_backend=Backend())
        try:
            for db in (baseline, shadow):
                db.add(user_id="a", request_id="r", session_id="s", messages=MESSAGES)
            for query in ("tea", "prefers", "unknown"):
                reports = [[(r.id, r.content, r.view, r.score) for r in
                            db.search(user_id="a", query=query, top_k=100).results]
                           for db in (baseline, shadow)]
                self.assertEqual(reports[0], reports[1])
        finally:
            baseline.close()
            shadow.close()

    def test_disabled_never_loads_or_calls_backend_search_identical(self):
        cfg = RetrieverConfig(fact_backend="nonexistent:factory")
        backend = Backend(RuntimeError("must never run"))
        with patch("aml_retriever.retriever.load_fact_backend", side_effect=AssertionError):
            db = RetrieverDB(cfg, fact_backend=backend)
        try:
            result = db.add(user_id="a", request_id="r", session_id="s", messages=MESSAGES)
            self.assertEqual(result.fact_status, "disabled")
            self.assertEqual(backend.calls, [])
            self.assertEqual(db.list_facts(user_id="a"), [])
        finally:
            db.close()

    def test_existing_raw_request_not_silently_backfilled_and_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "test.db")
            db = RetrieverDB(db_path=path)
            db.add(user_id="a", request_id="r", session_id="s", messages=MESSAGES)
            # Emulate the prior schema, not merely an empty current fact store.
            db._write(lambda con: con.executescript(
                "DROP TABLE fact_sources; DROP TABLE fact_extractions; DROP TABLE facts;"))
            db.close()
            backend = Backend()
            db = RetrieverDB(RetrieverConfig(db_path=path).with_flags(fact_extraction=True), fact_backend=backend)
            try:
                old = db.add(user_id="a", request_id="r", session_id="s", messages=MESSAGES)
                self.assertTrue(old.idempotent)
                self.assertEqual(old.fact_status, "disabled")
                self.assertEqual(backend.calls, [])
                db.add(user_id="a", request_id="r2", session_id="s", messages=MESSAGES)
            finally:
                db.close()
            db = RetrieverDB(db_path=path)
            try:
                self.assertEqual(len(db.list_facts(user_id="a")), 1)
                self.assertEqual(db.count("a"), 2)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
