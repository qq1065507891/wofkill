# -*- coding: utf-8 -*-
"""可见性策略：按观察者角色计算每个事实的可见性。

功能描述：认知管道的硬边界。任何下游模块不得超越此策略计算的可见性范围。
每条私密信息泄露必须可追溯至此处的策略规则。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-10
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact, StructuredWorldState


# ---------------------------------------------------------------------------
# Visibility data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactVisibility:
    """Visibility label for a single fact, computed by policy."""
    fact_index: int
    visibility: str
    audit_reason: str = ""


@dataclass
class VisibilityReport:
    """Full visibility computation result for audit."""
    viewer_id: str
    viewer_role: str
    visible_indices: list[int] = field(default_factory=list)
    hidden_indices: list[int] = field(default_factory=list)
    fact_labels: list[FactVisibility] = field(default_factory=list)
    leak_check_passed: bool = True
    leak_details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Visibility Policy
# ---------------------------------------------------------------------------

# Fact types and their default visibility
_FACT_VISIBILITY_MAP: dict[str, str] = {
    # Public facts
    "player_died": "public",
    "player_exiled": "public",
    "idiot_revealed": "public",
    "self_destruct": "public",
    "sheriff_elected": "public",
    "sheriff_registered": "public",
    "sheriff_withdraw": "public",
    "sheriff_vote_tie": "public",
    "sheriff_no_election": "public",
    "badge_transferred": "public",
    "badge_torn": "public",
    "speech": "public",
    "vote": "public",
    "claimed_role": "public",
    "claimed_suspect": "public",
    "claimed_good": "public",
    "seer_check_claim": "public",
    "badge_flow_claim": "public",

    # Wolf team private
    "wolf_kill_selected": "wolf_team",
    "wolf_no_kill_declared": "wolf_team",
    "wolf_no_kill_timeout": "wolf_team",
    "wolf_discussion": "wolf_team",

    # Seer private
    "seer_check": "seer_private",

    # Witch private
    "witch_antidote_used": "witch_private",
    "witch_poison_used": "witch_private",
    "witch_kill_target": "witch_private",

    # Hybrid private
    "hybrid_master_chosen": "hybrid_private",
}

# What each role can see
_ROLE_VISIBILITY: dict[str, set[str]] = {
    "villager": {"public"},
    "seer": {"public", "seer_private"},
    "witch": {"public", "witch_private"},
    "hunter": {"public", "hunter_private"},
    "idiot": {"public", "idiot_private"},
    "werewolf": {"public", "wolf_team"},
    "hybrid": {"public", "hybrid_private"},
}

# Forbidden categories — never leak to any player
_FORBIDDEN_FACT_TYPES: set[str] = {
    "moderator_note",
    "hidden_identity",
    "other_private_intent",
}

_OWNER_PRIVATE_FACT_TYPES: set[str] = {
    "seer_check",
    "hybrid_master_chosen",
}

_EVENT_VISIBILITY_TO_FACT_VISIBILITY = {
    "public": "public",
    "moderator_only": "moderator_only",
    "werewolf_team_only": "wolf_team",
    "seer_private": "seer_private",
    "witch_private": "witch_private",
    "hybrid_private": "hybrid_private",
}


class VisibilityPolicy:
    """Deterministic visibility computation per viewer role.

    Reads ruleset information_visibility config for authoritative visibility
    rules. Falls back to hardcoded defaults when config is absent.
    """

    # Class-level mirrors of module-level defaults. They exist so that
    # ``__init__`` can use ``hasattr(self, k)`` / ``setattr(self, k, ...)``
    # against canonical config keys (ruleset JSON uses the same uppercase
    # names) and override per-instance without mutating the module globals.
    _FACT_VISIBILITY_MAP: dict[str, str] = dict(_FACT_VISIBILITY_MAP)
    _ROLE_VISIBILITY: dict[str, set[str]] = {
        role: set(vis) for role, vis in _ROLE_VISIBILITY.items()
    }

    def __init__(self, visibility_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        if visibility_config:
            for k, v in visibility_config.items():
                if hasattr(self, k) and isinstance(v, dict):
                    # 深 merge：override 单个 fact_type
                    base = getattr(self, k, {})
                    merged = dict(base)
                    merged.update(v)
                    setattr(self, k, merged)
                elif hasattr(self, k):
                    setattr(self, k, v)
        # Sync the legacy private dicts to the (possibly overridden) class
        # attrs so the rest of the class (which still reads ``_fact_vis`` /
        # ``_role_vis``) sees the ruleset override.
        self._fact_vis = dict(self._FACT_VISIBILITY_MAP)
        self._role_vis = {role: set(vis) for role, vis in self._ROLE_VISIBILITY.items()}
        self._config = visibility_config or {}

    def compute_fact_visibility(
        self,
        fact_or_state,
        fact_index_or_viewer_role: Any = 0,
        viewer_role: str | None = None,
    ):
        """Compute visibility. Two supported signatures (polymorphic):

        - ``compute_fact_visibility(fact: StructuredFact, fact_index: int)``
          → returns a single :class:`FactVisibility` label (legacy API).
        - ``compute_fact_visibility(state: GameState, viewer_role: str)``
          → returns the list of :class:`StructuredFact` visible to that role
          (used by the ruleset-override audit test).
        """
        if isinstance(fact_or_state, StructuredFact):
            fact = fact_or_state
            fact_index = (
                fact_index_or_viewer_role
                if isinstance(fact_index_or_viewer_role, int)
                else 0
            )
            # Check forbidden first
            if fact.fact_type in _FORBIDDEN_FACT_TYPES:
                return FactVisibility(
                    fact_index=fact_index,
                    visibility="moderator_only",
                    audit_reason=f"forbidden fact type: {fact.fact_type}",
                )

            event_visibility = fact.metadata.get("visibility")
            if event_visibility in _EVENT_VISIBILITY_TO_FACT_VISIBILITY:
                mapped = _EVENT_VISIBILITY_TO_FACT_VISIBILITY[event_visibility]
                if mapped != "public":
                    return FactVisibility(
                        fact_index=fact_index,
                        visibility=mapped,
                        audit_reason=(
                            f"event visibility: {event_visibility} → {mapped}"
                        ),
                    )

            vis = self._fact_vis.get(fact.fact_type)
            if vis is not None:
                return FactVisibility(
                    fact_index=fact_index,
                    visibility=vis,
                    audit_reason=f"mapped fact_type: {fact.fact_type} → {vis}",
                )

            # Unknown fact types default to moderator-only. New event types must be
            # explicitly classified before they can enter player cognition.
            return FactVisibility(
                fact_index=fact_index,
                visibility="moderator_only",
                audit_reason=f"unmapped fact_type: {fact.fact_type}, default moderator_only",
            )

        # GameState path — build world state, filter for viewer role.
        from werewolf_agent.cognition.world_state import build_world_state
        role = viewer_role if viewer_role is not None else fact_index_or_viewer_role
        if not isinstance(role, str):
            raise TypeError(
                "compute_fact_visibility(state, ...) requires viewer_role (str); "
                f"got {type(role).__name__}"
            )
        ws = build_world_state(fact_or_state)
        # ``viewer_id`` is unused in role-based filtering; empty string keeps
        # audit reports unambiguous.
        return self.filter_visible_facts(ws, viewer_id="", viewer_role=role)

    def compute_visibility(
        self,
        world_state: StructuredWorldState,
        viewer_id: str,
        viewer_role: str,
    ) -> VisibilityReport:
        """Compute full visibility report for a viewer."""
        report = VisibilityReport(
            viewer_id=viewer_id,
            viewer_role=viewer_role,
        )

        allowed = self._role_vis.get(viewer_role, {"public"})

        for idx, fact in enumerate(world_state.facts):
            label = self.compute_fact_visibility(fact, idx)
            report.fact_labels.append(label)

            if label.visibility in allowed and self._viewer_owns_private_fact(
                fact,
                viewer_id,
            ):
                report.visible_indices.append(idx)
            else:
                report.hidden_indices.append(idx)

        return report

    @staticmethod
    def _viewer_owns_private_fact(fact: StructuredFact, viewer_id: str) -> bool:
        """限制单人私有事实只进入事实来源者视角。"""
        if fact.fact_type not in _OWNER_PRIVATE_FACT_TYPES:
            return True
        return bool(viewer_id) and fact.source_player == viewer_id

    def filter_visible_facts(
        self,
        world_state: StructuredWorldState,
        viewer_id: str,
        viewer_role: str,
    ) -> list[StructuredFact]:
        """Return only the facts visible to the given viewer."""
        report = self.compute_visibility(world_state, viewer_id, viewer_role)
        return [world_state.facts[i] for i in report.visible_indices]

    def check_no_leaks(
        self,
        world_state: StructuredWorldState,
        viewer_id: str,
        viewer_role: str,
        context_facts: list[StructuredFact],
    ) -> tuple[bool, list[str]]:
        """Verify that a set of context facts contains no visibility leaks.

        Returns (passed, leak_details).
        """
        report = self.compute_visibility(world_state, viewer_id, viewer_role)
        visible_set = set(report.visible_indices)
        leaks: list[str] = []

        fact_to_idx: dict[tuple[str, ...], int] = {}
        for idx, fact in enumerate(world_state.facts):
            key = (fact.fact_type, fact.source_player or "", fact.target_player or "", fact.value, fact.day)
            fact_to_idx[key] = idx
        for fact in context_facts:
            key = (fact.fact_type, fact.source_player or "", fact.target_player or "", fact.value, fact.day)
            if key not in fact_to_idx:
                continue
            idx = fact_to_idx[key]
            if idx not in visible_set:
                label = report.fact_labels[idx]
                leaks.append(
                    f"LEAK: fact {idx} ({fact.fact_type}) has visibility "
                    f"'{label.visibility}' but was included for {viewer_id} "
                    f"(role={viewer_role})"
                )

        return len(leaks) == 0, leaks
