# -*- coding: utf-8 -*-
"""
验证 ReflectionMemory repository 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/memory/test_reflection_repository.py -q
"""

from __future__ import annotations


def test_reflection_memory_remains_compatibly_importable() -> None:
    from werewolf_agent.memory import reflection
    from werewolf_agent.memory import reflection_repository

    assert reflection.ReflectionMemory is reflection_repository.ReflectionMemory
    assert reflection._LOG is reflection_repository._LOG
