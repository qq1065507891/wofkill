# -*- coding: utf-8 -*-
"""Live in-game cognition state for player agents.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
This module keeps per-viewer cognition matrices alive across turns. It is
separate from cross-game memory: the manager owns current-game belief updates,
while restored memory remains historical learning context.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from werewolf_agent.cognition.belief import BeliefState, BeliefUpdater
from werewolf_agent.cognition.claim_credibility import SeerClaimCredibilityEngine
from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import (
    StructuredFact,
    StructuredWorldState,
    extract_facts,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.memory.schemas import EvidenceItem
from werewolf_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class CognitionUpdateRecord:
    """Audit record for one viewer's cognition update."""

    viewer_id: str
    event_start: int
    event_end: int
    day: int
    phase: str
    deltas: list[dict[str, Any]] = field(default_factory=list)


class CognitionStateManager:
    """Owns live, visibility-safe cognition matrices for one game."""

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        *,
        visibility_policy: VisibilityPolicy | None = None,
    ) -> None:
        self.memory_store = memory_store or MemoryStore()
        self._visibility_policy = visibility_policy or VisibilityPolicy()
        self._belief_states: dict[str, BeliefState] = {}
        self._credibility_engines: dict[str, SeerClaimCredibilityEngine] = {}
        self._public_evidence: dict[str, PublicEvidenceIndex] = {}
        self._processed_event_count = 0
        self._role_names: list[str] = []
        self._player_ids: list[str] = []

    def initialize(self, game_state: GameState) -> None:
        """Create one uniform matrix and belief state per player."""
        self._player_ids = list(game_state.players.keys())
        self._role_names = list(dict.fromkeys(
            player.role
            for player in game_state.players.values()
            if player.role
        ))
        if not self._role_names:
            self._role_names = ["villager", "werewolf"]

        updater = BeliefUpdater(self._role_names)
        self._belief_states.clear()
        self._credibility_engines.clear()
        self._public_evidence.clear()
        self.memory_store.reset_game_memory()
        for viewer_id in self._player_ids:
            self.memory_store.init_matrix(
                viewer_id,
                self._player_ids,
                self._role_names,
            )
            self._belief_states[viewer_id] = updater.initialize(
                self._player_ids,
                viewer_id,
            )
            self._credibility_engines[viewer_id] = SeerClaimCredibilityEngine()
            self._public_evidence[viewer_id] = PublicEvidenceIndex()
        self._processed_event_count = 0

    def processed_event_count(self) -> int:
        return self._processed_event_count

    def update_from_events(self, game_state: GameState) -> list[CognitionUpdateRecord]:
        """Apply new events since the last update to all viewer matrices."""
        if not self._player_ids:
            self.initialize(game_state)

        event_start = self._processed_event_count
        event_end = len(game_state.events)
        if event_end <= event_start:
            return []

        world_state = self._world_state_for_events(game_state, event_start, event_end)
        updater = BeliefUpdater(self._role_names)
        records: list[CognitionUpdateRecord] = []

        for viewer_id in self._player_ids:
            player = game_state.players.get(viewer_id)
            if player is None:
                continue
            belief_state = self._belief_states.get(viewer_id)
            if belief_state is None:
                belief_state = updater.initialize(self._player_ids, viewer_id)
            engine = self._credibility_engines.get(viewer_id)
            if engine is None:
                engine = SeerClaimCredibilityEngine()
            evidence_index = self._public_evidence.get(viewer_id)
            if evidence_index is None:
                evidence_index = PublicEvidenceIndex()
            visible_facts = [
                fact for fact in self._visibility_policy.filter_visible_facts(
                    world_state,
                    viewer_id,
                    player.role,
                )
                if self._private_fact_allowed_for_viewer(fact, viewer_id)
            ]
            for fact in visible_facts:
                event_index = fact.metadata.get("source_event_index")
                if isinstance(event_index, int):
                    prefix = "claim" if fact.fact_type in {
                        "claimed_role", "claimed_good", "seer_check_claim"
                    } else "event"
                    evidence_index.observe_assignment_reference(
                        fact,
                        f"{prefix}:{game_state.game_id}:{event_index}",
                    )

            before = self._belief_snapshot(belief_state)
            belief_state = updater.update(
                belief_state,
                visible_facts,
                game_state.day_number,
                credibility=engine,
                public_evidence=evidence_index,
            )
            self._belief_states[viewer_id] = belief_state
            self._credibility_engines[viewer_id] = engine
            self._public_evidence[viewer_id] = evidence_index
            self.memory_store.sync_matrix(viewer_id, belief_state)
            evidence_refs = self._attach_evidence(viewer_id, visible_facts, game_state)
            after = self._belief_snapshot(belief_state)
            deltas = self._build_deltas(before, after, evidence_refs)
            if visible_facts or deltas:
                records.append(CognitionUpdateRecord(
                    viewer_id=viewer_id,
                    event_start=event_start,
                    event_end=event_end,
                    day=game_state.day_number,
                    phase=game_state.phase,
                    deltas=deltas,
                ))

        self._processed_event_count = event_end
        return records

    def prompt_belief_summary(
        self,
        viewer_id: str,
        game_state: GameState,
    ) -> dict[str, Any]:
        """Return a compact prompt-safe belief summary for one viewer."""
        matrix = self.memory_store.get_matrix(viewer_id)
        if matrix is None:
            return {}
        suspects: list[dict[str, Any]] = []
        trusted: list[dict[str, Any]] = []
        for entry in matrix.all_entries():
            player = game_state.players.get(entry.player_id)
            if player is None or not player.alive:
                continue
            top_role, top_prob = self._top_role(entry.role_probabilities)
            item = {
                "player": entry.player_id,
                "faction_lean": entry.faction_read,
                "trust": round(float(entry.trust), 2),
                "top_role_guess": top_role,
                "top_role_prob": round(float(top_prob), 2),
            }
            if entry.faction_read == "wolf_lean" or float(entry.trust) < 0.35:
                suspects.append(item)
            elif entry.faction_read == "good_lean" or float(entry.trust) > 0.65:
                trusted.append(item)
        return {
            "my_suspects": sorted(suspects, key=lambda x: x["trust"]),
            "my_trusted": sorted(trusted, key=lambda x: -x["trust"]),
        }

    def public_world_evidence(
        self,
        viewer_id: str,
    ) -> tuple[dict[str, dict[str, tuple[str, ...]]], set[str]]:
        """复用增量公开证据索引，避免上下文构建重扫事件。"""
        index = self._public_evidence.get(viewer_id)
        if index is None:
            return {}, set()
        return index.assignment_evidence(), index.assignment_evidence_ids()

    @staticmethod
    def _world_state_for_events(
        game_state: GameState,
        event_start: int,
        event_end: int,
    ) -> StructuredWorldState:
        world_state = StructuredWorldState()
        for event_index, event in enumerate(
            game_state.events[event_start:event_end],
            start=event_start,
        ):
            for fact in extract_facts(event, game_state):
                metadata = dict(fact.metadata)
                metadata["source_event_index"] = event_index
                world_state.append(replace(fact, metadata=metadata))
        return world_state

    @staticmethod
    def _private_fact_allowed_for_viewer(fact: StructuredFact, viewer_id: str) -> bool:
        if fact.fact_type == "seer_check":
            return fact.source_player == viewer_id
        if fact.fact_type == "hybrid_master_chosen":
            return fact.source_player == viewer_id
        return True

    @staticmethod
    def _belief_snapshot(belief_state: BeliefState) -> dict[str, tuple[str, float, str]]:
        snapshot: dict[str, tuple[str, float, str]] = {}
        for pid, belief in belief_state.beliefs.items():
            top_role, top_prob = belief.top_role_guess()
            snapshot[pid] = (top_role, round(top_prob, 4), belief.faction_lean)
        return snapshot

    @staticmethod
    def _top_role(role_probabilities: dict[str, float]) -> tuple[str, float]:
        if not role_probabilities:
            return ("unknown", 0.0)
        return max(role_probabilities.items(), key=lambda item: item[1])

    def _attach_evidence(
        self,
        viewer_id: str,
        facts: list[StructuredFact],
        game_state: GameState,
    ) -> dict[str, list[str]]:
        refs: dict[str, list[str]] = {}
        matrix = self.memory_store.get_matrix(viewer_id)
        if matrix is None:
            return refs
        for fact in facts:
            target_id = fact.target_player or fact.source_player
            if not target_id or target_id == viewer_id:
                continue
            claim = self._evidence_claim(fact)
            evidence = EvidenceItem(
                claim=claim,
                source_event=fact.fact_type,
                day=fact.day or game_state.day_number,
                confidence=0.6,
                speaker=fact.source_player,
            )
            matrix.add_evidence(target_id, evidence)
            refs.setdefault(target_id, []).append(claim)
        return refs

    @staticmethod
    def _evidence_claim(fact: StructuredFact) -> str:
        parts = [fact.fact_type]
        if fact.source_player:
            parts.append(f"source={fact.source_player}")
        if fact.target_player:
            parts.append(f"target={fact.target_player}")
        if fact.value:
            parts.append(f"value={fact.value}")
        return " ".join(parts)

    @staticmethod
    def _build_deltas(
        before: dict[str, tuple[str, float, str]],
        after: dict[str, tuple[str, float, str]],
        evidence_refs: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for player_id, after_value in after.items():
            before_value = before.get(player_id)
            if before_value != after_value or player_id in evidence_refs:
                deltas.append({
                    "target_id": player_id,
                    "before": before_value,
                    "after": after_value,
                    "evidence_refs": evidence_refs.get(player_id, []),
                })
        return deltas
