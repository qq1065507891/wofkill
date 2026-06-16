"""Replay artifact helpers for reproducible full-game ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ReplayRecord:
    trace_id: str
    output: dict[str, Any]
    event_index: int


@dataclass(frozen=True)
class ReplayArtifact:
    records: list[ReplayRecord] = field(default_factory=list)


class ReplayMatcher:
    def __init__(self, artifact: ReplayArtifact) -> None:
        self._artifact = artifact
        self.unsupported_reason = ""

    def match(
        self,
        trace_id: str,
        *,
        event_index: int,
        match_key: Literal["trace_id", "event_order"],
    ) -> ReplayRecord | None:
        self.unsupported_reason = ""
        if match_key == "trace_id":
            for record in self._artifact.records:
                if record.trace_id == trace_id:
                    return record
            self.unsupported_reason = "missing_replay_output"
            return None

        if match_key == "event_order":
            for record in self._artifact.records:
                if record.event_index == event_index:
                    return record
            self.unsupported_reason = "event_order_mismatch"
            return None

        self.unsupported_reason = f"unsupported_replay_match_key:{match_key}"
        return None
