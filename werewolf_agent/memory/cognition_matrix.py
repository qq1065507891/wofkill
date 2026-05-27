"""Short-term cognition matrix: per-player JSON-serializable state.

Each agent maintains a cognition matrix during a game with:
- role_probabilities per other player
- faction_read (good_lean / wolf_lean / unknown)
- trust score [0..1]
- key_evidence list (structured references to events)
- open_questions list

The matrix syncs from BeliefUpdater output but is the persisted form
that survives across turns and can be serialized to JSON.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.cognition.belief import BeliefState, PlayerBelief
from werewolf_agent.memory.schemas import CognitionMatrixEntry


class CognitionMatrix:
    """Per-viewer short-term cognition matrix for one game."""

    def __init__(self, viewer_id: str) -> None:
        self.viewer_id = viewer_id
        self._entries: dict[str, CognitionMatrixEntry] = {}

    def initialize(
        self,
        player_ids: list[str],
        role_names: list[str] | None = None,
    ) -> None:
        roles = role_names or [
            "villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid",
        ]
        if not roles:
            return
        uniform = 1.0 / len(roles)
        for pid in player_ids:
            if pid == self.viewer_id:
                continue
            self._entries[pid] = CognitionMatrixEntry(
                player_id=pid,
                role_probabilities={r: uniform for r in roles},
            )

    def get(self, player_id: str) -> CognitionMatrixEntry | None:
        return self._entries.get(player_id)

    def all_entries(self) -> list[CognitionMatrixEntry]:
        return list(self._entries.values())

    def update_from_belief(self, belief_state: BeliefState) -> None:
        """Sync from a BeliefUpdater BeliefState."""
        for pid, belief in belief_state.beliefs.items():
            entry = self._entries.get(pid)
            if entry is None:
                entry = CognitionMatrixEntry(player_id=pid)
                self._entries[pid] = entry
            entry.role_probabilities = dict(belief.role_probabilities)
            entry.faction_read = belief.faction_lean
            entry.trust = belief.trust
            entry.open_questions = list(belief.open_questions)

    def add_evidence(self, player_id: str, evidence: str) -> None:
        entry = self._entries.get(player_id)
        if entry is not None:
            entry.key_evidence.append(evidence)

    def add_open_question(self, player_id: str, question: str) -> None:
        entry = self._entries.get(player_id)
        if entry is not None:
            entry.open_questions.append(question)

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_id": self.viewer_id,
            "entries": {
                pid: entry.to_dict()
                for pid, entry in self._entries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitionMatrix":
        matrix = cls(viewer_id=data["viewer_id"])
        for pid, entry_data in data.get("entries", {}).items():
            matrix._entries[pid] = CognitionMatrixEntry.from_dict(entry_data)
        return matrix

    def player_ids(self) -> list[str]:
        return list(self._entries.keys())
