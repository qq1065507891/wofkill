"""Attention Filter: prune information by visibility + role.

Works with VisibilityPolicy to ensure only role-appropriate information
passes through. Adds role-specific filtering for information density.
"""

from __future__ import annotations

from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import StructuredFact, StructuredWorldState


class AttentionFilter:
    """Filters visible facts by role-specific attention rules.

    Depends on VisibilityPolicy for the hard visibility boundary.
    This layer adds soft pruning: even among visible facts, some may be
    less relevant for the current role's attention budget.
    """

    def __init__(self, visibility_policy: VisibilityPolicy) -> None:
        self._policy = visibility_policy

    def filter(
        self,
        world_state: StructuredWorldState,
        viewer_id: str,
        viewer_role: str,
        current_phase: str = "",
    ) -> list[StructuredFact]:
        """Return facts that are both visible AND attention-worthy."""
        visible = self._policy.filter_visible_facts(
            world_state, viewer_id, viewer_role,
        )
        return self._apply_role_attention(visible, viewer_role, current_phase)

    def _apply_role_attention(
        self,
        facts: list[StructuredFact],
        viewer_role: str,
        current_phase: str,
    ) -> list[StructuredFact]:
        """Apply role-specific attention rules to further prune facts."""
        # All visible facts pass through — the salience engine handles weighting.
        # This filter only removes facts that are explicitly irrelevant.
        # E.g., a villager's own status confirm doesn't need attention.
        result: list[StructuredFact] = []
        for fact in facts:
            if self._is_relevant(fact, viewer_role, current_phase):
                result.append(fact)
        return result

    @staticmethod
    def _is_relevant(
        fact: StructuredFact,
        viewer_role: str,
        current_phase: str,
    ) -> bool:
        """Check if a fact is relevant for the viewer's current attention."""
        # Speech with empty text is not relevant
        if fact.fact_type == "speech" and not fact.value.strip():
            return False
        return True
