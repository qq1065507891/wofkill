# -*- coding: utf-8 -*-
"""
验证 day finish 节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_day_finish_nodes.py -q
"""

from __future__ import annotations


def test_day_finish_nodes_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import day
    from werewolf_agent.runtime.nodes import day_finish

    assert day.check_victory is day_finish.check_victory
    assert day.finish_game is day_finish.finish_game
