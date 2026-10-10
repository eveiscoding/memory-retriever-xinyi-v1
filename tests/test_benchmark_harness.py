import json
import tempfile
import unittest
from pathlib import Path

from benchmark.adapters.locomo import load_locomo
from benchmark.chunking import adapter_word_count, chunk_messages
from benchmark.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from benchmark.report import render_markdown
from benchmark.runner import run_cases
from benchmark.schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


class FakeClient:
    def __init__(self):
        self.added = []

    def add(self, payload):
        self.added.extend(payload["messages"])
        return {"success": True}

    def search(self, payload):
        wanted = "updated" if "latest" in payload["query"] else "alpha"
        return [
            {"id": "wrong", "content": "unrelated"},
            {"id": "right", "content": next(m["content"] for m in self.added if wanted in m["content"])},
        ]


class BenchmarkHarnessTest(unittest.TestCase):
    def test_chunking_honors_message_and_word_limits(self):
        messages = tuple(BenchmarkMessage(str(i), "user", "one two", i) for i in range(5))
        self.assertEqual([2, 2, 1], [len(chunk) for chunk in chunk_messages(
            messages, max_messages=2, max_words=10,
        )])
        self.assertEqual(4, adapter_word_count("two words 中文"))

    def test_metrics_match_evidence_inside_aggregate_views(self):
        results = [
            {"content": "noise"},
            {"content": "[user] Alpha fact. [assistant] Beta fact."},
        ]
        gold = ("Alpha fact.", "Beta fact.")
        self.assertEqual(1.0, recall_at_k(results, gold, 2))
        self.assertEqual(0.5, reciprocal_rank(results, gold))
        self.assertEqual(0.6309297535714575, ndcg_at_k(results, gold, 2))

    def test_runner_uses_http_shaped_add_and_search(self):
        case = BenchmarkCase(
            "case", "fixture",
            (BenchmarkSession("s1", (
                BenchmarkMessage("m1", "user", "alpha fact", 1),
                BenchmarkMessage("m2", "user", "status updated", 2),
            )),),
            (BenchmarkProbe("q1", "find alpha", "fact", ("alpha fact",)),
             BenchmarkProbe("q2", "latest status", "temporal", ("status updated",))),
        )
        report = run_cases([case], FakeClient(), run_id="test")
        self.assertEqual(2, report["overall"]["queries"])
        self.assertEqual(1.0, report["overall"]["recall@20"])
        self.assertEqual(0.5, report["overall"]["mrr"])
        self.assertIn("Retrieval benchmark report", render_markdown(report))

    def test_locomo_adapter_maps_dialog_evidence(self):
        fixture = [{
            "sample_id": "sample-1",
            "conversation": {
                "speaker_a": "Alice", "speaker_b": "Bob",
                "session_1_date_time": "01:00 PM on 08 May 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "d1", "text": "I like tea."},
                    {"speaker": "Bob", "dia_id": "d2", "text": "Noted."},
                ],
            },
            "qa": [{"question": "What does Alice like?", "answer": "tea",
                    "category": 1, "evidence": ["d1"]}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            cases = load_locomo(path)
        self.assertEqual("Alice: I like tea.", cases[0].probes[0].gold_evidence[0])
        self.assertEqual("user", cases[0].sessions[0].messages[0].role)


if __name__ == "__main__":
    unittest.main()
