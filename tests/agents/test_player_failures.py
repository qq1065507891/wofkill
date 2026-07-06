# -*- coding: utf-8 -*-
"""
验证 player failure helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_player_failures.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import ActionType, FallbackAction


def test_player_failure_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.agents import player
    from werewolf_agent.agents import player_failures

    assert player._fallback_reason is player_failures.fallback_reason
    assert player._categorize_failure_category is player_failures.categorize_failure_category


def test_fallback_reason_does_not_embed_vote_target() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason

    action = FallbackAction(
        action_type=ActionType.VOTE,
        target_id="p07",
        speech="",
        reason="",
    )

    assert "p07" not in fallback_reason(action)
