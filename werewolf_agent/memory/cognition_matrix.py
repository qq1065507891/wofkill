# -*- coding: utf-8 -*-
"""
功能描述：短期认知矩阵——管理每局游戏中各玩家对其他玩家的角色概率、阵营判断和信任度。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.cognition.belief import BeliefState
from werewolf_agent.memory.schemas import CognitionMatrixEntry, EvidenceItem


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

    def add_evidence(
        self,
        player_id: str,
        evidence: EvidenceItem | str,
    ) -> None:
        """Add evidence for ``player_id``.

        MEM-07: accept either an ``EvidenceItem`` (preferred — carries
        source_event / day / confidence / speaker) or a bare ``str``
        (legacy back-compat — wrapped into EvidenceItem with default
        fields).

        MEM-NEW-9: explicit type guard. The pre-fix Union[EvidenceItem,
        str] silently wrapped everything else via
        ``EvidenceItem(claim=str(evidence))`` — including dicts, ints,
        and None — producing garbled evidence entries. Reject any
        other type with TypeError so the bug surfaces at the call
        site, not deep inside a downstream consumer that tries to
        read ``claim`` / ``source_event`` off a None.
        """
        if not isinstance(evidence, (EvidenceItem, str)):
            raise TypeError(
                f"CognitionMatrix.add_evidence: evidence must be "
                f"EvidenceItem or str, got {type(evidence).__name__}: "
                f"{evidence!r}"
            )
        entry = self._entries.get(player_id)
        if entry is None:
            return
        if isinstance(evidence, EvidenceItem):
            entry.key_evidence.append(evidence)
        else:
            # Back-compat: wrap a bare string claim into an
            # EvidenceItem with conservative defaults.
            entry.key_evidence.append(EvidenceItem(claim=evidence))

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
