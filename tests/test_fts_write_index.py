"""Small synthetic fixtures for indexed FTS replacement; no timing assertions."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aml_retriever.config import RetrieverConfig
from aml_retriever.retriever import RetrieverDB


class TestFTSWriteIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "fixture.db")
        self.db = RetrieverDB(RetrieverConfig(db_path=self.path))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def add(self, user="u", request="r", text="synthetic target"):
        return self.db.add(request_id=request, user_id=user, session_id="s",
                           messages=[{"content": text}])

    def assert_mapping(self):
        actual = [tuple(r) for r in self.db.query(
            "SELECT rowid,doc_id,user_id FROM fts ORDER BY rowid")]
        mapped = [tuple(r) for r in self.db.query(
            "SELECT fts_rowid,doc_id,user_id FROM fts_doc_rows ORDER BY fts_rowid")]
        self.assertEqual(actual, mapped)

    def test_append_update_and_idempotency(self):
        self.add()
        self.add(request="next", text="synthetic updated")
        self.assertTrue(self.add(request="next", text="synthetic updated").idempotent)
        self.assert_mapping()
        self.assertEqual(self.db.count(), 2)
        self.assertTrue(self.db.search(user_id="u", query="updated", top_k=10).results)

    def test_replacement_keeps_legacy_append_rowid_semantics(self):
        self.db._write(self.db._index_doc, "target", "u", "message", "old")
        self.db._write(self.db._index_doc, "other", "v", "message", "other")
        old = self.db.query("SELECT rowid FROM fts WHERE doc_id='target'")[0][0]
        self.db._write(self.db._index_doc, "target", "u", "message", "new")
        rows = self.db.query("SELECT rowid FROM fts WHERE doc_id='target'")
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][0], old)
        self.assert_mapping()

    def test_delete_reuse_and_purge_all(self):
        self.add()
        self.add(user="v")
        self.db.delete_user("u")
        self.assert_mapping()
        self.add(request="fresh")
        self.db.delete_user("v")
        self.assertEqual(self.db.count("u"), 1)
        self.assert_mapping()
        self.db.purge_all()
        self.assert_mapping()
        self.assertEqual(self.db.query("SELECT COUNT(*) FROM fts_doc_rows")[0][0], 0)

    def test_ttl_removes_mapping(self):
        self.add()
        self.add(user="v")
        self.db._write(lambda c: c.execute(
            "UPDATE messages SET added_at='2000-01-01T00:00:00+00:00' WHERE user_id='u'"))
        self.assertEqual(self.db.purge_stale_users(30)["deleted_users"], 1)
        self.assert_mapping()
        self.assertEqual(self.db.count("v"), 1)

    def test_failed_add_rolls_back_mapping_and_fts(self):
        self.add()
        before = [tuple(r) for r in self.db.query("SELECT rowid,* FROM fts ORDER BY rowid")]
        with patch.object(self.db, "_embed_doc", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                self.add(request="failed")
        self.assert_mapping()
        self.assertEqual(before, [tuple(r) for r in self.db.query(
            "SELECT rowid,* FROM fts ORDER BY rowid")])
        self.assertEqual(self.db.count(), 1)

    def test_legacy_migration_preserves_rows_duplicates_and_reopen(self):
        self.add()
        self.db.close()
        with sqlite3.connect(self.path) as con:
            con.execute("DROP TABLE fts_doc_rows")
            con.execute("INSERT INTO fts(text,doc_id,user_id,doc_type) "
                        "VALUES('legacy','duplicate','u','message')")
            con.execute("INSERT INTO fts(text,doc_id,user_id,doc_type) "
                        "VALUES('legacy2','duplicate','u','message')")
            old = con.execute("SELECT rowid,* FROM fts ORDER BY rowid").fetchall()
        self.db = RetrieverDB(RetrieverConfig(db_path=self.path))
        self.assertEqual(old, [tuple(r) for r in self.db.query("SELECT rowid,* FROM fts ORDER BY rowid")])
        self.assert_mapping()
        self.db._write(self.db._index_doc, "duplicate", "u", "message", "replacement")
        self.assertEqual(len(self.db.query("SELECT rowid FROM fts WHERE doc_id='duplicate'")), 1)
        self.db.close()
        self.db = RetrieverDB(RetrieverConfig(db_path=self.path))
        self.assert_mapping()

    def test_hot_delete_uses_mapping_and_rowid_not_doc_scan(self):
        plan = " ".join(r[3] for r in self.db.query(
            "EXPLAIN QUERY PLAN DELETE FROM fts WHERE rowid IN "
            "(SELECT fts_rowid FROM fts_doc_rows WHERE doc_id=?)", ("fixture",)))
        self.assertIn("idx_fts_doc_rows_doc", plan)
        self.assertIn("INDEX 0:=", plan)

    def test_mapping_failure_rolls_back_entire_add(self):
        self.add()
        before = [tuple(r) for r in self.db.query("SELECT rowid,* FROM fts ORDER BY rowid")]
        self.db._write(lambda c: c.execute(
            "CREATE TRIGGER synthetic_fail BEFORE INSERT ON fts_doc_rows "
            "BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.add(request="mapping-failure")
        self.assert_mapping()
        self.assertEqual(before, [tuple(r) for r in self.db.query("SELECT rowid,* FROM fts ORDER BY rowid")])
        self.assertEqual(self.db.query("SELECT COUNT(*) FROM requests WHERE request_id='mapping-failure'")[0][0], 0)

    def test_backfill_failure_is_atomic_and_retryable(self):
        self.add()
        self.db.close()
        with sqlite3.connect(self.path) as con:
            con.execute("DROP TABLE fts_doc_rows")
            # A legacy NULL violates the new map contract; no half-migrated map
            # may survive. No supported writer creates such rows.
            con.execute("INSERT INTO fts(text,doc_id,user_id,doc_type) VALUES('x',NULL,'u','message')")
        with self.assertRaises(sqlite3.IntegrityError):
            RetrieverDB(RetrieverConfig(db_path=self.path))
        with sqlite3.connect(self.path) as con:
            self.assertIsNone(con.execute("SELECT name FROM sqlite_master WHERE name='fts_doc_rows'").fetchone())
            con.execute("DELETE FROM fts WHERE doc_id IS NULL")
        self.db = RetrieverDB(RetrieverConfig(db_path=self.path))
        self.assert_mapping()


if __name__ == "__main__":
    unittest.main()
