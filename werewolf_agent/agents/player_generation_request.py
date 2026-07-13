# -*- coding: utf-8 -*-
"""
构建 PlayerAgent 生成请求，并把 provider 最终 system 观察器接入模型路由。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-13

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
from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import AgentContext, RetryInfo
from werewolf_agent.model_gateway.structured_output import (
    StructuredOutputMode,
)
from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly


@dataclass(frozen=True)
class PlayerGenerationRequest:
    """一次玩家动作生成尝试所需的模型请求参数。"""

    prompt: str
    system_prompt: str
    messages: tuple[dict[str, str], ...]
    persona_confirmed_in_system: bool
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
    prompt = agent._build_prompt(context, retry)
    system_prompt = agent._build_system_prompt(context)
    persona_text = _render_persona_text(agent, context)
    messages = (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    )
    request = PlayerGenerationRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        persona_confirmed_in_system=bool(
            persona_text and persona_text in system_prompt
        ),
        tools=tools,
        tool_choice=tool_choice,
        tool_call_required=tool_choice is not None,
        structured_output_mode=active_structured_mode.value,
    )
    _record_final_persona_proof(context, retry, request, persona_text)
    return request


def _record_final_persona_proof(
    context: AgentContext,
    retry: RetryInfo,
    request: PlayerGenerationRequest,
    persona_text: str,
) -> None:
    """只在最终 messages 已完整组装后确认 persona 的 system 注入。"""
    identity = getattr(context, "decision_identity", None)
    collector = getattr(context, "exposure_collector", None)
    if identity is None or collector is None or not context.persona_snapshot:
        return
    collector.record_persona_request_assembly(
        identity,
        request.messages,
        persona_text,
        _persona_attempt_kind(retry),
        attempt_ordinal=retry.attempt,
    )


def _render_persona_text(agent: Any, context: AgentContext) -> str:
    if not context.persona_snapshot:
        return ""
    persona_builder = getattr(agent, "_build_persona_prompt", None)
    if persona_builder is not None:
        return str(persona_builder(context) or "")
    return PlayerPromptBuilder(
        context,
        getattr(agent, "player_name", context.agent_id),
    )._build_persona()


def _persona_attempt_kind(retry: RetryInfo) -> str:
    if retry.attempt <= 1:
        return "initial"
    error_code = str(retry.error_code or "").lower()
    structured_markers = ("tool", "structured", "schema", "json", "parse")
    if any(marker in error_code for marker in structured_markers):
        return "structured_retry"
    return "semantic_retry"


def call_player_generation_request(
    agent: Any,
    context: AgentContext,
    request: PlayerGenerationRequest,
    generation_attempt_context: Any | None = None,
) -> Any:
    """使用已构建的请求调用模型路由。"""
    persona_text = _render_persona_text(agent, context)
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
        final_prompt_observer=_build_provider_persona_observer(context, persona_text),
    )


def _build_provider_persona_observer(
    context: AgentContext,
    persona_text: str,
) -> Any | None:
    """为本次玩家请求创建 provider 最终 system 观察回调。"""
    identity = getattr(context, "decision_identity", None)
    collector = getattr(context, "exposure_collector", None)
    if identity is None or collector is None or not context.persona_snapshot:
        return None

    def _observe(assembly: FinalPromptAssembly) -> None:
        collector.record_provider_persona_prompt_proof(
            identity,
            assembly.system_bytes,
            persona_text,
            assembly.attempt_kind,
            attempt_ordinal=assembly.attempt_ordinal,
            provider=assembly.provider,
            model=assembly.model,
            final_system_location=assembly.final_system_location,
            final_system_message_index=assembly.final_system_message_index,
        )

    return _observe
