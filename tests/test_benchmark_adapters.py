import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.adapters import (
    load_beam,
    load_clbench,
    load_halumem,
    load_locomo_refined,
    load_longmemeval,
    load_personamem,
    load_scriptmem,
)
from benchmark.adapters.common import parse_maybe_literal, timestamp_ms


class PublicAdapterTest(unittest.TestCase):
    def write_json(self, root, name, payload):
        path = Path(root) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_common_helpers(self):
        self.assertEqual(["a"], parse_maybe_literal('["a"]'))
        self.assertEqual(1_704_067_200_000, timestamp_ms("1704067200", 0))

    def test_longmemeval_custom_dataset_label(self):
        fixture = [{
            "question_id": "q", "question_type": "fact", "question": "Where?",
            "haystack_session_ids": ["s"], "haystack_dates": [""],
            "haystack_sessions": [[{"role": "user", "content": "In Paris.",
                                     "has_answer": True}]],
            "answer_session_ids": ["s"],
        }]
        with tempfile.TemporaryDirectory() as root:
            cases = load_longmemeval(self.write_json(root, "long.json", fixture),
                                     dataset="longmemeval-fixture")
        self.assertEqual("longmemeval-fixture", cases[0].dataset)

    def test_beam(self):
        fixture = [{
            "conversation_id": "beam-1", "chat": [{"turns": [[
                {"role": "user", "content": "I prefer tea.", "time_anchor": "2024-01-01"},
                {"role": "assistant", "content": "Noted."},
            ]]}],
            "probing_questions": repr({"preference": [{
                "question_text": "What drink is preferred?", "options": ["tea", "coffee"],
            }]}),
        }]
        with tempfile.TemporaryDirectory() as root:
            cases = load_beam(self.write_json(root, "beam.json", fixture))
        self.assertEqual("beam-preference", cases[0].probes[0].capability)
        self.assertEqual(("tea", "coffee"), cases[0].probes[0].options)
        self.assertEqual(2, len(cases[0].sessions[0].messages))

    def test_clbench(self):
        fixture = [{
            "idx": "cl-1", "context": "The deployment region is Singapore.",
            "question": "Where is it deployed?", "options": ["Singapore", "Tokyo"],
            "metadata": {"context_category": "deployment"},
        }]
        with tempfile.TemporaryDirectory() as root:
            cases = load_clbench(self.write_json(root, "cl.json", fixture))
        self.assertEqual("clbench-deployment", cases[0].probes[0].capability)

    def test_halumem_evidence(self):
        fixture = [{"uuid": "h-1", "sessions": [{
            "start_time": "2024-01-01T00:00:00Z",
            "dialogue": [{"role": "user", "content": "My bike is blue."}],
            "memory_points": [{"index": 7, "memory_content": "bike is blue",
                               "memory_source": "primary"}],
            "questions": [{"question": "What color is the bike?",
                           "question_type": "qa", "evidence": [7]}],
        }]}]
        with tempfile.TemporaryDirectory() as root:
            cases = load_halumem(self.write_json(root, "halu.json", fixture))
        self.assertEqual((("bike is blue",),), cases[0].probes[0].gold_evidence_groups)

    def test_halumem_embedded_evidence_object(self):
        fixture = [{"uuid": "h-2", "sessions": [{
            "dialogue": [{"role": "user", "content": "I started climbing."}],
            "questions": [{"question": "What sport?", "evidence": [{
                "memory_content": "started climbing", "memory_type": "Event Memory",
            }]}],
        }]}]
        with tempfile.TemporaryDirectory() as root:
            cases = load_halumem(self.write_json(root, "halu.json", fixture))
        self.assertEqual((("started climbing",),), cases[0].probes[0].gold_evidence_groups)

    def test_locomo_refined_two_file_export(self):
        conversations = [{"sample_id": "lr-1", "sessions": [{
            "session_id": "s1", "date": "2024-01-01T00:00:00Z",
            "messages": [{"role": "user", "content": "I enjoy hiking."}],
        }]}]
        questions = [{"sample_id": "lr-1", "qa_id": "q1",
                      "question": "What activity is enjoyed?", "category": "preference",
                      "evidence": ["lr-1:0:0"],
                      "evidence_messages": [{"role": "user", "text": "I enjoy hiking."}]}]
        with tempfile.TemporaryDirectory() as root:
            conversations_path = self.write_json(root, "conversations.json", conversations)
            questions_path = self.write_json(root, "questions.json", questions)
            cases = load_locomo_refined(conversations_path, questions_path=questions_path)
        self.assertEqual("locomo-refined-preference", cases[0].probes[0].capability)
        self.assertEqual(("I enjoy hiking.",), cases[0].probes[0].gold_evidence)
        self.assertTrue(cases[0].metadata["public_evidence"])

    def test_personamem_list_history(self):
        with tempfile.TemporaryDirectory() as root:
            history_dir = Path(root) / "histories"
            history_dir.mkdir()
            self.write_json(history_dir, "chat_history_32k_persona1.json", [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "I avoid red meat."},
            ])
            csv_path = Path(root) / "benchmark.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "persona_id", "user_query", "correct_answer", "incorrect_answers",
                    "pref_type", "chat_history_32k_link",
                ))
                writer.writeheader()
                writer.writerow({
                    "persona_id": "1", "user_query": "Recommend dinner",
                    "correct_answer": "vegetarian curry",
                    "incorrect_answers": '["steak"]', "pref_type": "implicit",
                    "chat_history_32k_link": "chat_history_32k_persona1.json",
                })
            cases = load_personamem(csv_path, chat_history_dir=history_dir)
        self.assertEqual(("vegetarian curry", "steak"), cases[0].probes[0].options)

    def test_scriptmem_with_local_dialogue(self):
        fixture = [{
            "sample_id": "scene-1",
            "conversation": {"session_1": [
                {"role": "user", "text": "The key is under the mat.", "dia_id": "d1"},
            ]},
            "qa": [{"question": "Where is the key?", "qa_type": "single",
                    "option": ["under the mat", "on the desk"]}],
        }]
        with tempfile.TemporaryDirectory() as root:
            cases = load_scriptmem(self.write_json(root, "angry.json", fixture))
        self.assertEqual("scriptmem-single", cases[0].probes[0].capability)

    def test_scriptmem_rejects_question_only_public_export(self):
        fixture = [{"conversation": {"format_example": True}, "qa": [{"question": "Q?"}]}]
        with tempfile.TemporaryDirectory() as root:
            path = self.write_json(root, "angry.json", fixture)
            with self.assertRaisesRegex(ValueError, "omit the copyrighted source dialogue"):
                load_scriptmem(path)


if __name__ == "__main__":
    unittest.main()
