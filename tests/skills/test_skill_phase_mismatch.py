"""P0: SkillDefinition.is_applicable() always returns False due to phase value mismatch.

SKILL.md frontmatter `applicable_phases` lists task-type values
('speech', 'night_action', 'sheriff_speech', etc.). But call sites
in prompt_builder.py:199 and context.py:403 pass `AgentContext.phase`
which is 'day' or 'night'. The intersection is always empty.

This is a latent bug — skill catalog is empty in production.

Fix: is_applicable should also accept task_type as a parameter.
When phase is 'day' or 'night', also match against task_type.
"""

from __future__ import annotations

from werewolf_agent.skills.schemas import SkillDefinition, SkillFaction, SkillName


def _make_skill(
    name: SkillName = SkillName.BOLD_CLAIM,
    applicable_roles: list[str] | None = None,
    applicable_phases: list[str] | None = None,
    faction: SkillFaction = SkillFaction.WOLF,
) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        display_name=name.value,
        description="test",
        applicable_roles=list(applicable_roles) if applicable_roles is not None else ["werewolf"],
        applicable_phases=list(applicable_phases) if applicable_phases is not None else ["speech"],
        faction=faction,
    )


def test_is_applicable_with_phase_day_matches_speech_skill():
    """Day phase + speech skill should be applicable."""
    skill = _make_skill(applicable_phases=["speech", "sheriff_speech"])
    assert skill.is_applicable("werewolf", phase="day", task_type="speech") is True


def test_is_applicable_with_phase_day_no_task_type_does_not_match_task_type_phases():
    """Without task_type, day phase should NOT match task-type-only phases.

    This documents the bug: original code only checks phase.
    """
    skill = _make_skill(applicable_phases=["speech"])
    # Without task_type, the original bug: phase='day' never matches 'speech'
    assert skill.is_applicable("werewolf", phase="day", task_type="") is False
    # But with task_type='speech' it should match
    assert skill.is_applicable("werewolf", phase="day", task_type="speech") is True


def test_is_applicable_with_phase_night_matches_night_action_skill():
    """Night phase + night_action skill should be applicable."""
    skill = _make_skill(applicable_phases=["night_action", "wolf_discussion"])
    assert skill.is_applicable("werewolf", phase="night", task_type="night_action") is True


def test_is_applicable_role_filter_still_works():
    """Role filter should still gate applicability."""
    skill = _make_skill(
        applicable_roles=["werewolf"],
        applicable_phases=["speech"],
    )
    # Werewolf in day + speech: applicable
    assert skill.is_applicable("werewolf", phase="day", task_type="speech") is True
    # Villager in day + speech: NOT applicable (role filter)
    assert skill.is_applicable("villager", phase="day", task_type="speech") is False


def test_is_applicable_empty_phases_list_means_always_applicable():
    """If applicable_phases is empty, the skill applies in any phase/task."""
    skill = _make_skill(applicable_phases=[])
    assert skill.is_applicable("werewolf", phase="day", task_type="speech") is True
    assert skill.is_applicable("werewolf", phase="night", task_type="night_action") is True


def test_prompt_builder_skill_analysis_hints_rendered_for_speech_role():
    """End-to-end: a villager in day/SPEECH phase with analyses gets the section.

    This is the regression test for the production bug. After P0-K1 the
    tool-path catalog is gone — the analyses are delivered via the
    `skill_analysis_hints` pre-injection section. We inject a hint and
    assert the section header is rendered.
    """
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.schemas import (
        ActionType,
        AgentContext,
        RetryInfo,
        TaskType,
    )
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        skill_analysis_hints={"wolf_pit": "嫌疑区: p05"},
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The pre-injection section is rendered when analyses are present.
    assert "技能分析结果" in prompt
    assert "嫌疑区" in prompt
