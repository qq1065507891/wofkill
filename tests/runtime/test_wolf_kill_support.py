# -*- coding: utf-8 -*-
"""
验证狼刀支持函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_wolf_kill_support.py -q
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState


def test_wolf_kill_support_exports_are_compatibility_imports() -> None:
    from werewolf_agent.runtime import agent_adapter
    from werewolf_agent.runtime import wolf_kill_support
    from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS

    assert agent_adapter._build_wolf_kill_directive is wolf_kill_support._build_wolf_kill_directive
    assert agent_adapter._single_wolf_vote is wolf_kill_support._single_wolf_vote
    assert agent_adapter.AGENT_TIMEOUTS is AGENT_TIMEOUTS


def test_build_wolf_kill_directive_remains_available_from_split_module() -> None:
    from werewolf_agent.runtime.wolf_kill_support import _build_wolf_kill_directive

    gs = GameState(
        game_id="wolf_kill_support",
        night_number=1,
        players={
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "v1": PlayerState(id="v1", role="villager", alive=True),
        },
    )

    directive = _build_wolf_kill_directive(gs, wolf_id="w1", plan={})

    assert isinstance(directive, str)
    assert directive
