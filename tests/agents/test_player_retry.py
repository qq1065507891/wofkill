# -*- coding: utf-8 -*-
"""
验证 player retry helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_player_retry.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType


def test_player_retry_helper_detects_repeated_signature() -> None:
    from werewolf_agent.agents.player_retry import check_repeat_error_signature

    retry = RetryInfo(error_code="parse_error")
    first_should_stop, signature = check_repeat_error_signature(
        retry,
        "not json",
        1,
        None,
        structured_output_mode="json_object",
    )
    second_should_stop, signature = check_repeat_error_signature(
        retry,
        "not json",
        2,
        signature,
        structured_output_mode="json_object",
    )

    assert first_should_stop is False
    assert second_should_stop is True
    assert retry.early_exit_reason == "repeat_error_signature: parse_error on attempts 1 and 2"


def test_player_retry_builds_vote_fallback_from_explicit_strategy_target() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p05"],
        strategy_directive={"_vote_fallback_target": "p05"},
    )

    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: "",
    )

    assert fallback.action_type == ActionType.VOTE
    assert fallback.target_id == "p05"
    assert "p05" not in fallback.reason


def test_player_retry_vote_fallback_uses_seer_claim_evidence() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p05"],
        salience_items=[
            {"type": "seer_claim", "target": "p02", "alignment": "werewolf"},
        ],
    )

    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: "",
    )

    assert fallback.target_id == "p02"


def test_player_retry_speech_fallback_uses_injected_speech_builder() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
    )

    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: "狼队夜间兜底发言",
    )

    assert fallback.speech == "狼队夜间兜底发言"
