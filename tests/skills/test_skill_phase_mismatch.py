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

from pathlib import Path

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


# ---------------------------------------------------------------------------
# S-01: applies_to_task_types is read from YAML frontmatter.
# ---------------------------------------------------------------------------

def test_applies_to_task_types_loaded_from_yaml(tmp_path):
    """S-01: `applies_to_task_types` declared in SKILL.md frontmatter must
    populate `SkillDefinition.applies_to_task_types` on load.

    Bug: the dataclass field exists, but `_load_manifests` did not read
    the YAML key — so any SKILL.md declaring the field would still load
    as an empty list. The P0-K2 precise filter would never fire.

    We use a temp skill directory and call `_load_manifests(root=...)`
    (a small refactor that lets the test inject the scan root).
    """
    from werewolf_agent.skills import werewolf_skills

    # Build a fresh skill directory with a SKILL.md that declares
    # applies_to_task_types: [speech].
    skill_dir = tmp_path / "test_skill_s01"
    skill_dir.mkdir()
    skill_md_content = (
        "---\n"
        "name: bold_claim\n"
        "display_name: 悍跳\n"
        "description: test\n"
        "applicable_roles:\n"
        "  - werewolf\n"
        "applicable_phases:\n"
        "  - speech\n"
        "applies_to_task_types:\n"
        "  - speech\n"
        "faction: wolf\n"
        "tags:\n"
        "  - test\n"
        "---\n"
    )
    (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")

    # Call the real loader pointed at our temp dir.
    loaded = werewolf_skills._load_manifests(root=tmp_path)
    assert len(loaded) == 1, f"expected exactly 1 loaded skill, got {len(loaded)}"
    skill = loaded[0]
    assert skill.applies_to_task_types == ["speech"], (
        f"S-01 regression: applies_to_task_types from YAML must be loaded; "
        f"got {skill.applies_to_task_types!r}"
    )

    # And end-to-end: the precise task-type filter actually fires.
    assert skill.is_applicable("werewolf", phase="speech", task_type="speech") is True
    assert skill.is_applicable("werewolf", phase="speech", task_type="vote") is False


# ---------------------------------------------------------------------------
# S-12: idiot is in find_power and hide_identity applicable_roles.
# ---------------------------------------------------------------------------

def test_idiot_in_find_power_and_hide_identity():
    """S-12: find_power and hide_identity both excluded idiot from
    applicable_roles. Idiot is a role that benefits from both skills
    (the role itself needs to be hidden, and the find_power signal
    can include the post-白露光 idiot as a known good target).
    """
    import yaml as _yaml

    # Load the YAML manifests directly.  The manifests live next to
    # the werewolf_skills.py module under `skills/manifests/`.
    from pathlib import Path as _Path
    from werewolf_agent.skills import werewolf_skills as _ws
    manifest_dir = _Path(_ws.__file__).parent / "manifests"
    fp_path = manifest_dir / "find_power.yaml"
    hi_path = manifest_dir / "hide_identity.yaml"
    fp_data = _yaml.safe_load(fp_path.read_text(encoding="utf-8")) or {}
    hi_data = _yaml.safe_load(hi_path.read_text(encoding="utf-8")) or {}

    # Both must include 'idiot' in applicable_roles.
    assert "idiot" in fp_data.get("applicable_roles", []), (
        f"S-12: find_power.yaml should include 'idiot' in applicable_roles; "
        f"got: {fp_data.get('applicable_roles')!r}"
    )
    assert "idiot" in hi_data.get("applicable_roles", []), (
        f"S-12: hide_identity.yaml should include 'idiot' in applicable_roles; "
        f"got: {hi_data.get('applicable_roles')!r}"
    )

    # And verify the loaded SkillDefinition picks up the new role list
    # (idiot becomes applicable).
    from werewolf_agent.skills.werewolf_skills import SKILL_DEFINITIONS
    from werewolf_agent.skills.schemas import SkillName
    find_def = next(d for d in SKILL_DEFINITIONS if d.name == SkillName.FIND_POWER)
    hide_def = next(d for d in SKILL_DEFINITIONS if d.name == SkillName.HIDE_IDENTITY)
    assert find_def.is_applicable("idiot", phase="speech", task_type="speech"), (
        "S-12: FIND_POWER manifest must apply to 'idiot' role"
    )
    assert hide_def.is_applicable("idiot", phase="speech", task_type="speech"), (
        "S-12: HIDE_IDENTITY manifest must apply to 'idiot' role"
    )
