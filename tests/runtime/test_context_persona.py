# -*- coding: utf-8 -*-
"""
验证 context persona 辅助函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_context_persona.py -q
"""

from __future__ import annotations


def test_persona_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime import context
    from werewolf_agent.runtime import context_persona

    assert context._load_persona_profile is context_persona._load_persona_profile
    assert context._get_persona_speech_style is context_persona._get_persona_speech_style
    assert context._get_persona_task_style is context_persona._get_persona_task_style
    assert context._SPEECH_STYLE_HINTS is context_persona._SPEECH_STYLE_HINTS
    assert context._SHERIFF_SPEECH_STYLE_OVERRIDES is context_persona._SHERIFF_SPEECH_STYLE_OVERRIDES
    assert context._TASK_STYLE_HINTS is context_persona._TASK_STYLE_HINTS
