# -*- coding: utf-8 -*-
"""
构建 PlayerAgent 每次生成尝试的模型请求。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.player_generation_request import build_player_generation_request
    >>> request = build_player_generation_request(agent, context, retry, mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from werewolf_agent.agents.player_generation import (
    generate_player_response as _generate_player_response,
)
from werewolf_agent.agents.schemas import AgentContext, RetryInfo
from werewolf_agent.model_gateway.structured_output import (
    StructuredOutputMode,
)


@dataclass(frozen=True)
class PlayerGenerationRequest:
    """一次玩家动作生成尝试所需的模型请求参数。"""

    prompt: str
    system_prompt: str
    tools: list[dict[str, Any]]
    tool_choice: dict[str, str] | None
    tool_call_required: bool
    structured_output_mode: str


def build_player_generation_request(
    agent: Any,
    context: AgentContext,
    retry: RetryInfo,
    active_structured_mode: StructuredOutputMode,
) -> PlayerGenerationRequest:
    """从当前 retry 状态和结构化模式构建模型请求。"""
    tools = [agent._player_action_tool(context)]
    tool_choice = (
        {"type": "tool", "name": "submit_player_action"}
        if active_structured_mode == StructuredOutputMode.NATIVE_TOOL
        else None
    )
    return PlayerGenerationRequest(
        prompt=agent._build_prompt(context, retry),
        system_prompt=agent._build_system_prompt(context),
        tools=tools,
        tool_choice=tool_choice,
        tool_call_required=tool_choice is not None,
        structured_output_mode=active_structured_mode.value,
    )


def call_player_generation_request(
    agent: Any,
    context: AgentContext,
    request: PlayerGenerationRequest,
    generation_attempt_context: Any | None = None,
) -> Any:
    """使用已构建的请求调用模型路由。"""
    return _generate_player_response(
        agent.model_router,
        agent_id=agent.agent_id,
        task_type=context.task_type.value,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        tools=request.tools,
        tool_choice=request.tool_choice,
        structured_output_mode=request.structured_output_mode,
        generation_attempt_context=generation_attempt_context,
    )
