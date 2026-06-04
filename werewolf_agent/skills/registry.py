"""Skill Registry: register, look up, and dispatch skills.

Agents query the registry for applicable skills based on role and phase.
The registry does not execute LLM calls — it returns structured skill
suggestions that the agent integrates into its decision process.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillInput,
    SkillName,
    SkillOutput,
)
from werewolf_agent.skills.werewolf_skills import (
    SKILL_DEFINITIONS,
    apply_skill,
)


# Roles that belong to the good faction
_GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}
_WOLF_ROLES = {"werewolf"}


def faction_for_role(role: str, gs: Any | None = None) -> SkillFaction:
    """Return the faction a role belongs to.

    Most roles are statically mapped (villager→GOOD, werewolf→WOLF).
    Hybrid's faction is dynamic — it depends on its master's faction
    (S-02): if `gs.hybrid_master_faction` is "werewolf", hybrid is
    WOLF-aligned; otherwise it falls back to GOOD. If `gs` is None
    (test seam), we conservatively return GOOD.
    """
    if role in _WOLF_ROLES:
        return SkillFaction.WOLF
    if role == "hybrid":
        # S-02: dispatch on master's faction when known.
        if gs is not None and getattr(gs, "hybrid_master_faction", None) == "werewolf":
            return SkillFaction.WOLF
        return SkillFaction.GOOD
    return SkillFaction.GOOD


class SkillRegistry:
    """Central registry for werewolf agent skills."""

    def __init__(self) -> None:
        self._skills: dict[SkillName, SkillDefinition] = {}
        for skill in SKILL_DEFINITIONS:
            self._skills[skill.name] = skill

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: SkillName) -> SkillDefinition | None:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def count(self) -> int:
        return len(self._skills)

    def applicable_skills(self, role: str, phase: str) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if s.is_applicable(role, phase)]

    def dispatch(self, name: SkillName, skill_input: SkillInput) -> SkillOutput:
        """Apply a skill and return structured output."""
        return apply_skill(name, skill_input)

    def dispatch_applicable(
        self,
        skill_input: SkillInput,
    ) -> list[SkillOutput]:
        """Dispatch all applicable skills for the given input."""
        applicable = self.applicable_skills(skill_input.role, skill_input.phase)
        return [
            self.dispatch(s.name, skill_input)
            for s in applicable
        ]

    def by_role(self, role: str) -> list[SkillDefinition]:
        return [
            s for s in self._skills.values()
            if not s.applicable_roles or role in s.applicable_roles
        ]

    def by_phase(self, phase: str) -> list[SkillDefinition]:
        return [
            s for s in self._skills.values()
            if not s.applicable_phases or phase in s.applicable_phases
        ]

    def by_tag(self, tag: str) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if tag in s.tags]

    def by_faction(self, faction: SkillFaction) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if s.faction == faction]

    def skills_for_role(self, role: str, gs: Any | None = None) -> list[SkillDefinition]:
        """Return skills available to a role based on its faction.

        Loading rule:
        - WOLF roles get: WOLF + COMMON + UNIVERSAL
        - GOOD roles get: GOOD + COMMON + UNIVERSAL

        For hybrid, the resolved faction depends on its master's faction
        (see S-02). Pass `gs` to enable that branch; without `gs`,
        hybrid falls back to GOOD.
        """
        role_faction = faction_for_role(role, gs=gs)
        allowed = {SkillFaction.COMMON, SkillFaction.UNIVERSAL}
        allowed.add(role_faction)
        return [
            s for s in self._skills.values()
            if s.faction in allowed
            and (not s.applicable_roles or role in s.applicable_roles)
        ]

    def dispatch_for_role(
        self,
        role: str,
        phase: str,
        skill_input: SkillInput,
        task_type: str = "",
        gs: Any | None = None,
    ) -> list[SkillOutput]:
        """Dispatch all faction-applicable skills for a role in a given phase.

        P0-K2: when `task_type` is provided, the dispatch is filtered by
        `SkillDefinition.applies_to_task_types` (in addition to the
        existing `applicable_phases` / `applicable_roles` checks).

        S-02: when `gs` is provided, hybrid's faction is resolved from
        `gs.hybrid_master_faction` (default: GOOD). When hybrid's master
        is a werewolf, hybrid is treated as `werewolf` for the role
        filter on WOLF-faction skills — the manifest role list still
        gates the dispatch, but the gate is opened for the wolf-aligned
        hybrid.
        """
        role_faction = faction_for_role(role, gs=gs)
        # S-02: when hybrid is wolf-aligned, treat it as `werewolf` for
        # the role filter so WOLF-faction skills (e.g. bold_claim,
        # swing_vote) are reachable.
        effective_role = role
        if role == "hybrid" and role_faction == SkillFaction.WOLF:
            effective_role = "werewolf"
        allowed = {SkillFaction.COMMON, SkillFaction.UNIVERSAL}
        allowed.add(role_faction)
        skills = [
            s for s in self._skills.values()
            if s.faction in allowed
            and s.is_applicable(effective_role, phase, task_type=task_type)
        ]
        return [self.dispatch(s.name, skill_input) for s in skills]

    def names(self) -> list[str]:
        return [s.name.value for s in self._skills.values()]
