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
    SkillOutput,
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


def test_markdown_body_append_respects_final_prompt_cap() -> None:
    from werewolf_agent.skills.werewolf_skills import PROMPT_INJECTABLE_CAP

    registry = SkillRegistry()
    registry.register(SkillDefinition(
        name=SkillName.BOLD_CLAIM,
        display_name="bold_claim",
        description="test",
        applicable_roles=["werewolf"],
        applicable_phases=["speech"],
        applies_to_task_types=["speech"],
        faction=SkillFaction.WOLF,
        body="超长技能正文" * 500,
    ))

    outputs = registry.dispatch_for_role(
        role="werewolf",
        phase="speech",
        task_type="speech",
        skill_input=SkillInput(
            role="werewolf",
            phase="speech",
            task_type="speech",
        ),
    )
    output = next(out for out in outputs if out.skill_name == "bold_claim")

    assert len(output.prompt_injectable) <= PROMPT_INJECTABLE_CAP
    assert output.prompt_injectable.endswith("...（已省略）")


def test_dispatch_for_role_records_each_skill_success_and_failure() -> None:
    """P2-2: skill 调用监控必须逐个记录名称、输入摘要、成功/失败和结果摘要。"""

    class AuditedRegistry(SkillRegistry):
        def __init__(self) -> None:
            self._skills = {
                SkillName.PUSH_VOTE: SkillDefinition(
                    name=SkillName.PUSH_VOTE,
                    display_name="归票",
                    description="push vote",
                    faction=SkillFaction.COMMON,
                    applicable_phases=["vote"],
                ),
                SkillName.RESIST_PUSH: SkillDefinition(
                    name=SkillName.RESIST_PUSH,
                    display_name="抗推",
                    description="resist push",
                    faction=SkillFaction.COMMON,
                    applicable_phases=["vote"],
                ),
            }

        def dispatch(self, name: SkillName, skill_input: SkillInput) -> SkillOutput:
            if name == SkillName.RESIST_PUSH:
                raise RuntimeError("boom private wolf")
            return SkillOutput(
                skill_name=name.value,
                confidence=0.82,
                prompt_injectable="建议对p02归票。",
                reasoning="p02发言矛盾",
                risk_alerts=["不要过度跟票"],
                metadata={"evidence_refs": ["event:1"]},
            )

    records: list[dict[str, object]] = []
    outputs = AuditedRegistry().dispatch_for_role(
        "villager",
        "vote",
        SkillInput(
            role="villager",
            phase="vote",
            task_type="vote",
            day=1,
            legal_targets=["p02", "p03"],
            extra={"wolf_team_plan": {"private": "hidden"}},
        ),
        task_type="vote",
        audit_records=records,
    )

    assert [output.skill_name for output in outputs] == ["push_vote"]
    assert [record["call_name"] for record in records] == ["push_vote", "resist_push"]
    assert records[0]["status"] == "success"
    assert records[0]["success"] is True
    assert records[0]["result_available_to_decision"] is True
    assert records[0]["decision_usage"] == "prompt_injected"
    assert records[0]["input_summary"] == {
        "role": "villager",
        "phase": "vote",
        "task_type": "vote",
        "day": 1,
        "legal_target_count": 2,
        "has_wolf_team_plan": True,
    }
    assert records[0]["output_summary"] == {
        "confidence": 0.82,
        "has_prompt_injectable": True,
        "risk_alert_count": 1,
        "evidence_ref_count": 1,
    }
    assert records[1]["status"] == "error"
    assert records[1]["success"] is False
    assert records[1]["error_type"] == "RuntimeError"
    assert "boom" in str(records[1]["error_message"])


def test_skill_bodies_do_not_reintroduce_false_role_rules() -> None:
    registry = SkillRegistry()
    bodies = {
        skill.name.value: skill.body
        for skill in registry.all_skills()
    }

    assert "预言家查验你是好人" not in bodies["deep_hook"]
    assert "狼人不敢刀你" not in bodies["counter_claim"]
    assert "查杀更有说服力" not in bodies["bold_claim"]
    assert "被放逐对白痴有利" not in bodies["resist_push"]
    assert "不需抗推" not in bodies["resist_push"]
    assert "死亡方式不证明遗言内容为真" in bodies["last_words"]


def test_shared_power_analysis_is_private_for_good_roles() -> None:
    registry = SkillRegistry()
    find_power = registry.get(SkillName.FIND_POWER)
    assert find_power is not None
    assert "仅用于私下防守分析" in find_power.body
    assert "不得在公开发言中点明" in find_power.body
