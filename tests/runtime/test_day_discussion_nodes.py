# -*- coding: utf-8 -*-
"""
验证 day discussion 节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_day_discussion_nodes.py -q
"""

from __future__ import annotations

import importlib


def test_day_discussion_node_remains_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import day
    from werewolf_agent.runtime.nodes import day_discussion

    assert day.free_discussion is day_discussion.free_discussion


def test_day_facade_monkeypatch_propagates_discussion_dependencies(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import day

    day_discussion = importlib.import_module("werewolf_agent.runtime.nodes.day_discussion")

    def fake_dispatch(*args, **kwargs):
        return None

    def fake_pick_order(*args, **kwargs):
        return []

    monkeypatch.setattr(day, "_dispatch_agent", fake_dispatch)
    monkeypatch.setattr(day, "agent_sheriff_pick_speech_order", fake_pick_order)

    assert day_discussion._dispatch_agent is fake_dispatch
    assert day_discussion.agent_sheriff_pick_speech_order is fake_pick_order
