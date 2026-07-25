# -*- coding: utf-8 -*-
"""推理策略最低等级与玩家配置启动校验测试。"""

import pytest


@pytest.mark.parametrize(
    ("task_type", "minimum"),
    [
        ("speech", "medium"),
        ("last_words", "medium"),
        ("vote", "medium"),
        ("sheriff_speech", "medium"),
        ("discussion_summary", "medium"),
        ("wolf_team_plan", "high"),
        ("wolf_discussion", "high"),
        ("deception", "high"),
        ("reflection", "high"),
        ("night_action", "high"),
        ("hunter_shot", "high"),
    ],
)
def test_every_player_task_has_required_minimum(task_type, minimum):
    from werewolf_agent.model_gateway.reasoning_policy import minimum_reasoning_level

    assert minimum_reasoning_level(task_type).value == minimum


def test_unknown_player_llm_task_fails_closed():
    from werewolf_agent.model_gateway.reasoning_policy import minimum_reasoning_level

    with pytest.raises(ValueError, match="unknown player LLM task"):
        minimum_reasoning_level("mystery_player_task")


@pytest.mark.parametrize("configured", [None, "none", "low"])
def test_validation_names_profile_below_speech_minimum(configured):
    from werewolf_agent.model_gateway.reasoning_policy import validate_player_reasoning_profiles

    model = {"provider": "openai", "model": "x"}
    if configured is not None:
        model["reasoning"] = {"level": configured}
    with pytest.raises(ValueError, match="weak_profile"):
        validate_player_reasoning_profiles(
            model_profiles={"weak_model": model},
            llm_profiles={"weak_profile": {"default": {"provider": "openai", "model_profile": "weak_model"}}},
            player_assignments={"p01": "weak_profile"},
        )


def test_high_task_validation_reports_required_and_actual_levels():
    from werewolf_agent.model_gateway.reasoning_policy import validate_player_reasoning_profiles

    with pytest.raises(
        ValueError,
        match=r"weak_profile.*task 'deception'.*required 'high'.*actual 'medium'",
    ):
        validate_player_reasoning_profiles(
            model_profiles={"medium": {"provider": "openai", "model": "m", "reasoning": {"level": "medium"}}},
            llm_profiles={"weak_profile": {"default": {"provider": "openai", "model_profile": "medium"}}},
            player_assignments={"p01": "weak_profile"},
        )


def test_glm_compatible_adapter_is_not_treated_as_reasoning_capable():
    from werewolf_agent.model_gateway.reasoning_policy import validate_player_reasoning_profiles

    with pytest.raises(ValueError, match="glm_profile"):
        validate_player_reasoning_profiles(
            model_profiles={"glm": {"provider": "glm", "model": "glm", "reasoning": {"level": "high"}}},
            llm_profiles={"glm_profile": {"default": {"provider": "glm", "model_profile": "glm"}}},
            player_assignments={"p01": "glm_profile"},
        )
