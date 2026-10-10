"""Dataset-neutral schema for AML Add/Search evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkMessage:
    message_id: str
    role: str
    content: str
    timestamp: int

    def as_api_message(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


@dataclass(frozen=True)
class BenchmarkSession:
    session_id: str
    messages: tuple[BenchmarkMessage, ...]


@dataclass(frozen=True)
class BenchmarkProbe:
    probe_id: str
    query: str
    capability: str
    gold_evidence: tuple[str, ...] = ()
    forbidden_evidence: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    after_session: int | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dataset: str
    sessions: tuple[BenchmarkSession, ...]
    probes: tuple[BenchmarkProbe, ...]
    metadata: dict = field(default_factory=dict, compare=False)

    def validate(self) -> None:
        if not self.case_id or not self.dataset:
            raise ValueError("case_id and dataset are required")
        if not self.sessions or not self.probes:
            raise ValueError(f"{self.case_id}: sessions and probes must be non-empty")
        session_ids = [session.session_id for session in self.sessions]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError(f"{self.case_id}: duplicate session_id")
        message_ids = [m.message_id for s in self.sessions for m in s.messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError(f"{self.case_id}: duplicate message_id")
        for probe in self.probes:
            if probe.after_session is not None and not 0 <= probe.after_session < len(self.sessions):
                raise ValueError(f"{probe.probe_id}: after_session is out of range")
