# -*- coding: utf-8 -*-
"""
封装 PlayerAgent 的模型生成调用和最近失败原因读取。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.agents.player_generation import generate_player_response
    >>> generate_player_response(router, agent_id="p01", task_type="vote", prompt="...")
"""

from __future__ import annotations

import inspect
from typing import Any


def generate_player_response(
    model_router: Any,
    *,
    agent_id: str,
    task_type: str,
    prompt: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    tool_choice: dict[str, Any] | None,
    structured_output_mode: str,
    generation_attempt_context: Any | None = None,
    final_prompt_observer: Any | None = None,
) -> Any:
    kwargs = dict(
        agent_id=agent_id,
        task_type=task_type,
        prompt=prompt,
        system_prompt=system_prompt,
        tools=tools,
        tool_choice=tool_choice,
        structured_output_mode=structured_output_mode,
    )
    if generation_attempt_context is not None:
        kwargs["generation_attempt_context"] = generation_attempt_context
    router_parameters = inspect.signature(model_router.generate).parameters
    accepts_extra_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in router_parameters.values()
    )
    if final_prompt_observer is not None and (
        "final_prompt_observer" in router_parameters or accepts_extra_kwargs
    ):
        kwargs["final_prompt_observer"] = final_prompt_observer
    return model_router.generate(**kwargs)


def latest_generation_failure_reason(model_router: Any) -> str | None:
    get_usage_log = getattr(model_router, "get_usage_log", None)
    if get_usage_log is None:
        return None
    usage_log = get_usage_log()
    if not usage_log:
        return None
    last_record = usage_log[-1]
    if last_record.success or not last_record.fallback_reason:
        return None
    return str(last_record.fallback_reason)
