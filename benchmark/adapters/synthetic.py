"""Bridge the repository's deterministic proxy set into the HTTP harness."""
from __future__ import annotations

from aml_retriever.evaluation.dataset import make_dataset

from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def load_synthetic(*, seed: int = 20260806, scale: str = "smoke",
                   difficulty: str = "mixed", suite: str = "v11") -> list[BenchmarkCase]:
    dataset = make_dataset(seed=seed, scale=scale, difficulty=difficulty, suite=suite)
    content_lookup = {
        f"{session['session_id']}#{index}": message["content"]
        for session in dataset.sessions
        for index, message in enumerate(session["messages"])
    }
    cases: list[BenchmarkCase] = []
    for user_id in dataset.users:
        user_sessions = [session for session in dataset.sessions if session["user_id"] == user_id]
        sessions = tuple(
            BenchmarkSession(
                session_id=session["session_id"],
                messages=tuple(
                    BenchmarkMessage(
                        message_id=f"{session['session_id']}#{index}",
                        role=message["role"],
                        content=message["content"],
                        timestamp=message["timestamp"],
                    )
                    for index, message in enumerate(session["messages"])
                ),
            )
            for session in user_sessions
        )
        probes = tuple(
            BenchmarkProbe(
                probe_id=query.qid,
                query=query.text,
                capability=query.kind,
                gold_evidence=tuple(content_lookup[item] for item in query.gold),
                forbidden_evidence=tuple(content_lookup[item] for item in query.distractors),
            )
            for query in dataset.queries
            if query.user_id == user_id
        )
        case = BenchmarkCase(user_id, "synthetic-v11", sessions, probes)
        case.validate()
        cases.append(case)
    return cases
