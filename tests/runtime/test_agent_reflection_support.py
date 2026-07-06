# -*- coding: utf-8 -*-
"""
验证赛后反思支持函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_agent_reflection_support.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import AgentContext, TaskType


def test_reflection_support_exports_are_compatibility_imports() -> None:
    from werewolf_agent.runtime import agent_adapter
    from werewolf_agent.runtime import agent_reflection_support

    assert agent_adapter._strip_in_game_directives is agent_reflection_support._strip_in_game_directives
    assert agent_adapter._agent_reflection is agent_reflection_support._agent_reflection


def test_strip_in_game_directives_available_from_split_module() -> None:
    from werewolf_agent.runtime.agent_reflection_support import _strip_in_game_directives

    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.REFLECTION,
        strategy_directive={
            "role_alerts": ["不要保留"],
            "reflection_task": "复盘任务",
            "game_outcome": "好人胜利",
        },
    )

    stripped = _strip_in_game_directives(context)

    assert stripped.strategy_directive == {
        "reflection_task": "复盘任务",
        "game_outcome": "好人胜利",
    }
