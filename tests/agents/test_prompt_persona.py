# -*- coding: utf-8 -*-
"""
验证 persona prompt 渲染拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_prompt_persona.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import AgentContext, RetryInfo, TaskType


def test_prompt_persona_methods_remain_compatibly_available() -> None:
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.prompt_persona import PromptPersonaMixin

    assert PlayerPromptBuilder._build_persona is PromptPersonaMixin._build_persona
    assert PlayerPromptBuilder._slim_numeric_params is PromptPersonaMixin._slim_numeric_params


def test_persona_is_rendered_only_in_final_system_prompt() -> None:
    """动态 persona 仍须成为最终 system 消息的一部分，且不得重复到 user。"""
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        persona_snapshot={"profile_id": "calm", "tone": "measured_unique_tone"},
    )
    builder = PlayerPromptBuilder(context)

    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(RetryInfo())

    assert "measured_unique_tone" in system_prompt
    assert "measured_unique_tone" not in user_prompt
