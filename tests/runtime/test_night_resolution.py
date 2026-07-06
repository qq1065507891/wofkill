# -*- coding: utf-8 -*-
"""
验证夜晚结算节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_night_resolution.py -q
"""

from __future__ import annotations


def test_night_resolution_node_remains_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import night
    from werewolf_agent.runtime.nodes import night_resolution

    assert night.resolve_night is night_resolution.resolve_night
