# -*- coding: utf-8 -*-
"""
验证夜晚入口节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_night_entry.py -q
"""

from __future__ import annotations


def test_night_entry_nodes_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import night
    from werewolf_agent.runtime.nodes import night_entry

    assert night.enter_night is night_entry.enter_night
    assert night.night_hunter_idiot_status is night_entry.night_hunter_idiot_status
