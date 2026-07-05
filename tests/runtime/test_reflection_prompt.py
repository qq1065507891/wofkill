# -*- coding: utf-8 -*-
"""
验证赛后复盘提示词会按角色阵营选择正确模板。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.reflection_prompt import build_reflection_prompt
    >>> build_reflection_prompt(player, winner="good", hybrid_master_faction=None)
"""

from types import SimpleNamespace

from werewolf_agent.runtime.reflection_prompt import build_reflection_prompt


def test_build_reflection_prompt_uses_good_template_for_good_role_win():
    prompt = build_reflection_prompt(
        SimpleNamespace(role="seer"),
        winner="good",
        hybrid_master_faction=None,
    )

    assert "你是seer,本局好人阵营胜" in prompt
    assert "【保留的优点】" in prompt
    assert "必须列出 1-2 个具体策略" in prompt


def test_build_reflection_prompt_uses_wolf_template_for_hybrid_wolf_master():
    prompt = build_reflection_prompt(
        SimpleNamespace(role="hybrid"),
        winner="werewolf",
        hybrid_master_faction="werewolf",
    )

    assert "你是狼人,本局狼队胜" in prompt
    assert "深水/冲锋/倒钩" in prompt
