"""Visibility Policy: compute per-fact visibility by viewer role.

This is the HARD BOUNDARY of the cognition pipeline. No downstream module
may expand visibility beyond what this policy computes. Every private
information leak must be traceable to a policy rule here.

Visibility levels:
- public: visible to all players
- wolf_team: visible only to werewolves (wolf discussion, night kill target)
- seer_private: visible only to seer (check results)
- witch_private: visible only to witch (wolf kill target, potion inventory)
- hybrid_private: visible only to hybrid (master id)
- hunter_private: visible only to hunter (shoot status)
- idiot_private: visible only to idiot (reveal status)
- moderator_only: never visible to any player agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact, StructuredWorldState
from werewolf_agent.core.models import GameState


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
    "claimed_claim": "public",
    "claimed_role": "public",
    "claimed_suspect": "public",
    "claimed_badge_flow": "public",

    # Wolf team private
    "wolf_kill_target": "wolf_team",
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

    # Hunter private
    "hunter_can_shoot": "hunter_private",

    # Idiot private
    "idiot_reveal_status": "idiot_private",
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


class VisibilityPolicy:
    """Deterministic visibility computation per viewer role.

    Reads ruleset information_visibility config for authoritative visibility
    rules. Falls back to hardcoded defaults when config is absent.
    """

    def __init__(self, visibility_config: dict[str, Any] | None = None) -> None:
        self._config = visibility_config or {}
        self._fact_vis = dict(_FACT_VISIBILITY_MAP)
        self._role_vis = dict(_ROLE_VISIBILITY)

    def compute_fact_visibility(
        self,
        fact: StructuredFact,
        fact_index: int,
    ) -> FactVisibility:
        """Compute visibility label for a single fact."""
        # Check forbidden first
        if fact.fact_type in _FORBIDDEN_FACT_TYPES:
            return FactVisibility(
                fact_index=fact_index,
                visibility="moderator_only",
                audit_reason=f"forbidden fact type: {fact.fact_type}",
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

            if label.visibility in allowed:
                report.visible_indices.append(idx)
            else:
                report.hidden_indices.append(idx)

        return report

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

        for fact in context_facts:
            if fact not in world_state.facts:
                continue
            idx = world_state.facts.index(fact)
            if idx not in visible_set:
                label = report.fact_labels[idx]
                leaks.append(
                    f"LEAK: fact {idx} ({fact.fact_type}) has visibility "
                    f"'{label.visibility}' but was included for {viewer_id} "
                    f"(role={viewer_role})"
                )

        return len(leaks) == 0, leaks
