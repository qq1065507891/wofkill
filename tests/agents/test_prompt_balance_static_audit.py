"""Static regression checks for player-facing prompt balance."""

from __future__ import annotations

from pathlib import Path

from werewolf_agent.agents.directive_priority import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
)
from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder, RetryInfo
from werewolf_agent.agents.schemas import AgentContext, TaskType


ROOT = Path(__file__).resolve().parents[2]

PROMPT_SURFACES = (
    ROOT / "werewolf_agent" / "agents" / "prompt_builder.py",
    ROOT / "werewolf_agent" / "runtime" / "agent_adapter.py",
    ROOT / "werewolf_agent" / "runtime" / "strategy" / "witch.py",
    ROOT / "werewolf_agent" / "runtime" / "directives" / "villager.py",
    ROOT / "werewolf_agent" / "runtime" / "directives" / "witch.py",
    ROOT / "config" / "personas" / "judge_profiles.yaml",
)


def test_known_false_or_absolute_prompt_phrases_are_absent() -> None:
    forbidden = (
        "首夜大概率应该救人",
        "首夜必须救人",
        "不救的风险极高",
        "投票时优先跟随查杀方",
        "跟随查杀方归票",
        "强烈建议使用毒药毒杀",
        "月光守护了所有人",
    )
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PROMPT_SURFACES
    )

    for phrase in forbidden:
        assert phrase not in text, phrase


def test_known_directive_keys_are_classified() -> None:
    expected_hard = {
        "must_address_alerts",
        "role_alerts",
        "vote_pressure",
        "gold_water_duty",
        "unreported_checks",
        "my_check_history",
        "vote_basis_hint",
    }
    expected_suggestions = {
        "seer_speech_directive",
        "witch_speech_directive",
        "hunter_speech_directive",
        "villager_speech_directive",
        "idiot_speech_directive",
        "hybrid_speech_directive",
        "wolf_speech_directive",
    }
    expected_reference = {
        "day_discussion_summary",
        "vote_pressure_context",
        "skill_tactical_advice",
    }

    assert expected_hard <= HARD_CONSTRAINT_KEYS
    assert expected_suggestions <= SUGGESTION_KEYS
    assert expected_reference <= REFERENCE_KEYS


def test_directive_priority_sets_do_not_overlap() -> None:
    assert not (HARD_CONSTRAINT_KEYS & SUGGESTION_KEYS)
    assert not (HARD_CONSTRAINT_KEYS & REFERENCE_KEYS)
    assert not (SUGGESTION_KEYS & REFERENCE_KEYS)


def test_system_prompt_defers_field_level_contract_to_dynamic_prompt() -> None:
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
    )
    builder = PlayerPromptBuilder(ctx)

    stable_contract = builder._build_output_contract()
    user_prompt = builder.build_user_prompt(RetryInfo())

    for discriminator in ("action_type", "choice", "intent"):
        assert discriminator not in stable_contract
    assert "ActionContract" in stable_contract
    assert "最终输出协议" in user_prompt


def test_compact_json_overflow_remains_valid_json() -> None:
    import json

    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
    )
    builder = PlayerPromptBuilder(ctx)
    big = {f"k{i}": "x" * 200 for i in range(50)}

    rendered = builder._compact_json(big)
    parsed = json.loads(rendered)

    assert parsed["truncated"] is True
    assert parsed["original_type"] == "dict"
    assert "content_prefix" in parsed or "head" in parsed


def test_static_audit_does_not_touch_wolf_team_plan_files() -> None:
    changed_surfaces = {
        path.relative_to(ROOT).as_posix()
        for path in PROMPT_SURFACES
    }

    assert not any("wolf_team_plan" in path for path in changed_surfaces)
