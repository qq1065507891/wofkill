# -*- coding: utf-8 -*-
"""
验证 ReflectionQualityGate 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/memory/test_reflection_quality.py -q
"""

from __future__ import annotations


def test_reflection_quality_gate_remains_compatibly_importable() -> None:
    from werewolf_agent.memory import reflection
    from werewolf_agent.memory import reflection_quality

    assert reflection.ReflectionQualityGate is reflection_quality.ReflectionQualityGate
