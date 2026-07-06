# -*- coding: utf-8 -*-
"""
验证 context skill advice 辅助函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_context_skill_advice.py -q
"""

from __future__ import annotations


def test_skill_advice_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime import context
    from werewolf_agent.runtime import context_skill_advice

    assert context._skill_output_to_advice_frame is context_skill_advice._skill_output_to_advice_frame
    assert context._skill_advice_frame_to_prompt_dict is context_skill_advice._skill_advice_frame_to_prompt_dict
    assert context._inject_skill_output is context_skill_advice._inject_skill_output
