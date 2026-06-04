"""P0-K2: Add `applies_to_task_types` to SkillDefinition for precise filtering.

Audit finding: `applicable_phases` mixes two distinct concepts:
- coarse phase ('day' / 'night')
- precise task type ('speech', 'night_action', 'sheriff_speech', etc.)

The new field `applies_to_task_types` is dedicated to precise task-type
filtering. `dispatch_for_role` should now accept and pass `task_type` to
`is_applicable`, and `is_applicable` should also honor the new field.

This is a regression test: it pins down the contract for the new field
and ensures the registry dispatches the right skills.
"""

from __future__ import annotations

import pytest

from werewolf_agent.skills.registry import SkillRegistry
from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillInput,
    SkillName,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    name: SkillName = SkillName.BOLD_CLAIM,
    applicable_roles: list[str] | None = None,
    applicable_phases: list[str] | None = None,
    applies_to_task_types: list[str] | None = None,
    faction: SkillFaction = SkillFaction.WOLF,
) -> SkillDefinition:
    """Build a SkillDefinition with the K2 fields."""
    return SkillDefinition(
        name=name,
        display_name=name.value,
        description="test",
        applicable_roles=list(applicable_roles) if applicable_roles is not None else ["werewolf"],
        applicable_phases=list(applicable_phases) if applicable_phases is not None else [],
        applies_to_task_types=list(applies_to_task_types) if applies_to_task_types is not None else [],
        faction=faction,
    )


# ---------------------------------------------------------------------------
# K2.1: SkillDefinition has the new field
# ---------------------------------------------------------------------------

class TestSkillDefinitionAppliesToTaskTypes:
    """The dataclass exposes a new `applies_to_task_types` field."""

    def test_field_default_empty(self):
        """A new SkillDefinition defaults to `applies_to_task_types=[]`."""
        skill = SkillDefinition(
            name=SkillName.BOLD_CLAIM,
            display_name="bold_claim",
            description="test",
        )
        assert hasattr(skill, "applies_to_task_types"), (
            "SkillDefinition must expose the new `applies_to_task_types` field."
        )
        assert skill.applies_to_task_types == []

    def test_field_assignable(self):
        """The new field is a list[str] and accepts explicit values."""
        skill = SkillDefinition(
            name=SkillName.BOLD_CLAIM,
            display_name="bold_claim",
            description="test",
            applies_to_task_types=["speech", "sheriff_speech"],
        )
        assert skill.applies_to_task_types == ["speech", "sheriff_speech"]


# ---------------------------------------------------------------------------
# K2.2: is_applicable honors the new field
# ---------------------------------------------------------------------------

class TestIsApplicableWithTaskTypes:
    """is_applicable must check `applies_to_task_types` when provided."""

    def test_empty_task_types_means_no_filter(self):
        """Empty list = no task-type filter (skill is gated only by role/phase)."""
        skill = _make_skill(
            applicable_roles=["werewolf"],
            applicable_phases=["speech"],
            applies_to_task_types=[],
        )
        # Both task types pass because the filter is empty.
        assert skill.is_applicable("werewolf", "speech", task_type="speech") is True
        assert skill.is_applicable("werewolf", "speech", task_type="vote") is True

    def test_task_type_in_list_passes(self):
        """When task_type is in the list, the skill is applicable."""
        skill = _make_skill(
            applicable_roles=["werewolf"],
            applies_to_task_types=["speech"],
        )
        assert skill.is_applicable("werewolf", "", task_type="speech") is True

    def test_task_type_not_in_list_fails(self):
        """When task_type is NOT in the list, the skill is filtered out."""
        skill = _make_skill(
            applicable_roles=["werewolf"],
            applies_to_task_types=["speech"],
        )
        # task_type="vote" is not in the list -> filtered out
        assert skill.is_applicable("werewolf", "", task_type="vote") is False

    def test_role_filter_still_applies_with_task_type_filter(self):
        """Role filter is enforced independently of task-type filter."""
        skill = _make_skill(
            applicable_roles=["werewolf"],
            applies_to_task_types=["speech"],
        )
        # Villager + matching task_type: still filtered by role
        assert skill.is_applicable("villager", "", task_type="speech") is False

    def test_phase_filter_still_applies_with_task_type_filter(self):
        """applicable_phases (coarse phase) still gates applicability.

        Note: applicable_phases is an OR check — either phase or task_type
        must match. applies_to_task_types is an AND check — task_type
        must be in the list. Both filters compose: all must pass.
        """
        # Case A: applicable_phases not satisfied at all (no match via OR)
        skill = _make_skill(
            applicable_roles=["werewolf"],
            applicable_phases=["night_action"],
            applies_to_task_types=["speech"],
        )
        # phase="vote" and task_type="vote" - neither matches
        # night_action and task_type="vote" is not in applies_to_task_types
        assert skill.is_applicable("werewolf", "vote", task_type="vote") is False

        # Case B: applicable_phases satisfied via task_type (OR), task_type filter satisfied
        skill_b = _make_skill(
            applicable_roles=["werewolf"],
            applicable_phases=["speech"],
            applies_to_task_types=["speech"],
        )
        # phase="vote" doesn't match, but task_type="speech" matches both filters
        assert skill_b.is_applicable("werewolf", "vote", task_type="speech") is True

        # Case C: applicable_phases satisfied via task_type, but applies_to_task_types NOT satisfied
        skill_c = _make_skill(
            applicable_roles=["werewolf"],
            applicable_phases=["speech"],
            applies_to_task_types=["night_action"],
        )
        # task_type="speech" matches applicable_phases but not applies_to_task_types
        assert skill_c.is_applicable("werewolf", "vote", task_type="speech") is False


# ---------------------------------------------------------------------------
# K2.3: dispatch_for_role filters by task_type
# ---------------------------------------------------------------------------

class TestDispatchForRoleFilteredByTaskType:
    """The registry's `dispatch_for_role` accepts a `task_type` parameter."""

    def test_dispatch_for_role_filters_by_task_type(self):
        """Skills with applies_to_task_types must be filtered out by task_type."""
        # Register a fresh registry and a single skill that ONLY applies to "speech".
        registry = SkillRegistry()
        speech_only = _make_skill(
            name=SkillName.BOLD_CLAIM,
            applicable_roles=["werewolf"],
            applicable_phases=["speech"],
            applies_to_task_types=["speech"],
        )
        registry.register(speech_only)

        # Build minimal SkillInput (dispatcher just calls apply_skill).
        skill_input = SkillInput(role="werewolf", phase="speech")

        # 1) matching task_type -> the skill IS dispatched
        outputs_match = registry.dispatch_for_role(
            role="werewolf", phase="speech", skill_input=skill_input,
            task_type="speech",
        )
        match_names = [o.skill_name for o in outputs_match]
        assert "bold_claim" in match_names, (
            "When task_type='speech' matches applies_to_task_types, "
            "the skill must be dispatched."
        )

        # 2) non-matching task_type -> the skill is NOT dispatched
        outputs_miss = registry.dispatch_for_role(
            role="werewolf", phase="speech", skill_input=skill_input,
            task_type="night_action",
        )
        miss_names = [o.skill_name for o in outputs_miss]
        assert "bold_claim" not in miss_names, (
            "When task_type='night_action' is NOT in applies_to_task_types, "
            "the skill must be filtered out."
        )

    def test_dispatch_for_role_without_task_type_param(self):
        """The function should still support no-task_type call sites."""
        registry = SkillRegistry()
        skill_no_filter = _make_skill(
            name=SkillName.BOLD_CLAIM,
            applicable_roles=["werewolf"],
            applies_to_task_types=[],
        )
        registry.register(skill_no_filter)

        skill_input = SkillInput(role="werewolf", phase="speech")
        # Default: no task_type given -> no task-type filter
        outputs = registry.dispatch_for_role(
            role="werewolf", phase="speech", skill_input=skill_input,
        )
        names = [o.skill_name for o in outputs]
        assert "bold_claim" in names
