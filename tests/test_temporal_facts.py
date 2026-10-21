import copy
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from aml_retriever.config import RetrieverConfig
from aml_retriever.retriever import RetrieverDB


class EchoFacts:
    identity = "synthetic-echo-fixture-not-quality"
    def extract(self, *, messages, **kwargs):
        return {"facts": [dict(text=m["content"], kind="profile", sources=[
            dict(message_index=m["message_index"], quote=m["content"])]) for m in messages]}


def make_db(**kwargs):
    return RetrieverDB(RetrieverConfig().with_flags(fact_extraction=True, fact_versions=True),
                       fact_backend=EchoFacts(), **kwargs)


def add_fact(db, text, user="a", role="user"):
    return db.add(request_id=text + role, user_id=user, session_id="s", messages=[
        dict(role=role, content=text, timestamp=1700000000000)]).fact_ids[0]


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.old = add_fact(self.db, "Works at Alpha")
        self.new = add_fact(self.db, "Works at Beta")

    def tearDown(self):
        self.db.close()

    def record(self, fid=None, **overrides):
        args = dict(user_id="a", fact_id=fid or self.old, operation_id="one", version_key="employment",
                    valid_at=100, invalid_at=None)
        args.update(overrides)
        with patch("aml_retriever.temporal_facts.now_ms", return_value=overrides.pop("clock", 1000)):
            args.pop("clock", None)
            return self.db.record_fact_version(**args)

    def ids(self, valid_at, known_at=10000):
        return [r["fact_id"] for r in self.db.facts_as_of(user_id="a", valid_at=valid_at, known_at=known_at)]

    def test_bitemporal_supersession_preserves_both_histories(self):
        old = self.record()
        new = self.record(self.new, operation_id="two", valid_at=200,
                          supersedes=old["version_id"], clock=2000)
        self.assertEqual(self.ids(99), [])
        self.assertEqual(self.ids(199), [self.old])
        self.assertEqual(self.ids(200), [self.new])
        self.assertEqual(self.ids(250, 1500), [self.old])
        self.assertEqual(self.ids(250, 999), [])
        self.assertEqual(self.ids(250, 2000), [self.new])
        stored = self.db.query("SELECT * FROM fact_versions WHERE version_id=?", (old["version_id"],))[0]
        self.assertEqual(stored["retired_at"], 2000)
        self.assertIsNone(stored["invalid_at"])
        self.assertEqual(new["supersedes"], old["version_id"])
        self.assertEqual(self.db.count("a"), 2)

    def test_future_validity_does_not_hide_current_fact(self):
        old = self.record()
        self.record(self.new, operation_id="future", valid_at=5000, supersedes=old["version_id"], clock=2000)
        self.assertEqual(self.ids(2500, 2500), [self.old])
        self.assertEqual(self.ids(5000, 2500), [self.new])

    def test_unknown_validity_is_not_inferred_from_message_or_record_time(self):
        self.record(valid_at=None)
        self.assertEqual(self.ids(1000), [])

    def test_explicit_end_is_exclusive(self):
        self.record(invalid_at=200)
        self.assertEqual(self.ids(100), [self.old])
        self.assertEqual(self.ids(199), [self.old])
        self.assertEqual(self.ids(200), [])

    def test_idempotency_and_changed_payload_conflict(self):
        first = self.record()
        self.assertEqual(first, self.record())
        with self.assertRaises(ValueError):
            self.record(valid_at=101)
        self.assertEqual(len(self.db.query("SELECT * FROM fact_versions")), 1)

    def test_overlapping_unspecified_update_rejected_atomically(self):
        self.record()
        before = [dict(r) for r in self.db.query("SELECT * FROM fact_versions")]
        with self.assertRaises(ValueError):
            self.record(self.new, operation_id="two", valid_at=200)
        self.assertEqual(before, [dict(r) for r in self.db.query("SELECT * FROM fact_versions")])

    def test_adjacent_intervals_need_no_inferred_conflict(self):
        self.record(invalid_at=200)
        self.record(self.new, operation_id="two", valid_at=200)
        self.assertEqual(self.ids(200), [self.new])

    def test_cross_user_predecessor_and_fact_rejected(self):
        old = self.record()
        bob = add_fact(self.db, "Works at Alpha", user="b")
        with self.assertRaises(ValueError):
            self.record(bob, operation_id="bad")
        with self.assertRaises(ValueError):
            self.record(bob, user_id="b", operation_id="bad", supersedes=old["version_id"], valid_at=200)
        self.assertEqual(self.db.facts_as_of(user_id="b", valid_at=150, known_at=1500), [])

    def test_different_subject_keys_and_roles_independent(self):
        self.record()
        self.record(self.new, operation_id="other-key", version_key="another-person:employment")
        assistant = add_fact(self.db, "Works at Alpha", role="assistant")
        self.record(assistant, operation_id="assistant")
        self.assertEqual(len(self.ids(150)), 3)

    def test_wrong_key_role_stale_or_out_of_order_predecessor_rejected(self):
        old = self.record()
        assistant = add_fact(self.db, "Works at Beta", role="assistant")
        with self.assertRaises(ValueError):
            self.record(assistant, operation_id="assistant-update", valid_at=200,
                        supersedes=old["version_id"])
        for kwargs in [dict(version_key="wrong"), dict(valid_at=100), dict(valid_at=None)]:
            with self.assertRaises(ValueError):
                self.record(self.new, operation_id="two", supersedes=old["version_id"], **kwargs)
        self.record(self.new, operation_id="two", supersedes=old["version_id"], valid_at=200, clock=2000)
        with self.assertRaises(ValueError):
            self.record(operation_id="three", supersedes=old["version_id"], valid_at=300, clock=3000)

    def test_invalid_time_types_and_intervals(self):
        for bad in [True, 1.5, "2026-01-01", 253402300800000]:
            with self.assertRaises(ValueError):
                self.record(valid_at=bad)
        for end in [99, 100]:
            with self.assertRaises(ValueError):
                self.record(invalid_at=end)
        with self.assertRaises(ValueError):
            self.db.facts_as_of(user_id="a", valid_at=True)

    def test_backwards_system_clock_rejected_without_changes(self):
        self.record(clock=2000)
        with self.assertRaises(ValueError):
            self.record(self.new, operation_id="two", version_key="another", clock=1999)
        self.assertEqual(len(self.db.query("SELECT * FROM fact_versions")), 1)

    def test_concurrent_same_operation_has_one_version(self):
        def once(_):
            return self.db.record_fact_version(user_id="a", fact_id=self.old, operation_id="same",
                version_key="employment", valid_at=100)
        with ThreadPoolExecutor(max_workers=8) as pool:
            versions = list(pool.map(once, range(16)))
        self.assertEqual(len({v["version_id"] for v in versions}), 1)
        self.assertEqual(len(self.db.query("SELECT * FROM fact_versions")), 1)

    def test_deletion_ttl_and_full_purge_include_history(self):
        self.record()
        self.db.delete_user("a")
        for table in ("fact_versions", "fact_version_ops"):
            self.assertEqual(len(self.db.query(f"SELECT * FROM {table}")), 0)
        self.old = add_fact(self.db, "New fact")
        self.record(operation_id="new")
        self.db._write(lambda con: con.execute("UPDATE messages SET added_at='2000-01-01'"))
        self.db.purge_stale_users(1)
        self.assertEqual(len(self.db.query("SELECT * FROM fact_versions")), 0)
        self.old = add_fact(self.db, "Last fact")
        self.record(operation_id="last")
        self.db.purge_all()
        self.assertEqual(len(self.db.query("SELECT * FROM fact_version_ops")), 0)

    def test_disabled_version_annotation_rejected(self):
        self.db.flags["fact_versions"] = False
        with self.assertRaises(ValueError):
            self.record()
        with self.assertRaises(ValueError):
            self.ids(100)

    def test_schema_restart_preserves_versions(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "test.db")
            db = make_db(db_path=path)
            try:
                fid = add_fact(db, "Persistent")
                db.record_fact_version(user_id="a", fact_id=fid, operation_id="op", version_key="key", valid_at=100)
            finally:
                db.close()
            db = make_db(db_path=path)
            try:
                self.assertEqual(db.facts_as_of(user_id="a", valid_at=100)[0]["fact_id"], fid)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
