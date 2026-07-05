# -*- coding: utf-8 -*-
"""
功能描述：**：从 player.py 拆出，负责决策使用哪种解析器并将上下文数据传递到底层实现。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.output_parser import (
    parse_choice_action as _parse_choice_impl,
    parse_speech_intent_action as _parse_speech_impl,
    uses_choice_pipeline as _uses_choice_impl,
    uses_speech_intent_pipeline as _uses_speech_impl,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    PlayerAction,
)


def select_output_mode(
    *,
    legal_actions: list[ActionType],
    legal_targets: list[str],
    task_type: Any,
    speech_intent_tasks: set,
) -> OutputMode:
    """Pick the simplest output mode that captures what the task needs.

    - TARGET_CHOICE: only one target-requiring action is legal and has targets
    - SPEECH_INTENT: speech task with pure SPEECH or legacy SPEECH+VOTE
    - FULL_ACTION: default (full PlayerAction schema)
    """
    if _uses_choice_impl(legal_actions, legal_targets):
        return OutputMode.TARGET_CHOICE
    if _uses_speech_impl(legal_actions, task_type, speech_intent_tasks):
        return OutputMode.SPEECH_INTENT
    return OutputMode.FULL_ACTION


def parse_choice_action(
    text: str,
    context: AgentContext,
) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
    """Parse a TARGET_CHOICE output into a PlayerAction + audit data."""
    return _parse_choice_impl(
        text,
        context.legal_actions,
        context.legal_targets,
        context.salience_items,
    )


def parse_speech_intent_action(
    text: str,
    context: AgentContext,
) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
    """Parse a SPEECH_INTENT output into a PlayerAction + audit data."""
    return _parse_speech_impl(
        text,
        context.agent_id,
        context.own_role,
        context.legal_targets,
        context.salience_items,
        context.visible_world_state,
        context.recent_transcript,
    )
