# -*- coding: utf-8 -*-
"""
测试防御性发言阶段的策略指令和兜底文案。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.defense_speech_directives import build_defense_context_directive
    >>> build_defense_context_directive()
"""

from werewolf_agent.runtime.defense_speech_directives import (
    build_defense_context_directive,
    build_empty_defense_speech_fallback,
)


def test_build_defense_context_directive_lists_response_requirements() -> None:
    """防御性发言指令要求直接回应指控并提供解释。"""
    directive = build_defense_context_directive()

    assert "你正处于被质疑/被指控的状态" in directive
    assert "直接回应针对你的具体指控" in directive
    assert "不要泛泛地喊'我真是好人'" in directive


def test_build_empty_defense_speech_fallback_keeps_existing_wording() -> None:
    """空防御发言兜底文案保持稳定。"""
    assert build_empty_defense_speech_fallback("p01") == (
        "我是p01，我理解大家的质疑。让我解释一下："
        "我当时的判断基于公开信息，可能不全面但绝不是恶意带节奏。"
        "请大家听完我的解释后再做决定。"
    )
