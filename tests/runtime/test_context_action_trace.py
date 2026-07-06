# -*- coding: utf-8 -*-
"""
验证 context action trace 辅助函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_context_action_trace.py -q
"""

from __future__ import annotations


def test_action_trace_payload_remains_compatibly_importable() -> None:
    from werewolf_agent.runtime import context
    from werewolf_agent.runtime import context_action_trace

    assert context._action_trace_payload is context_action_trace._action_trace_payload
