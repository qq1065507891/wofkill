# -*- coding: utf-8 -*-
"""
验证 day vote / exile 节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_day_vote_nodes.py -q
"""

from __future__ import annotations

import importlib


def test_day_vote_nodes_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import day

    day_vote_nodes = importlib.import_module("werewolf_agent.runtime.nodes.day_vote")

    assert day.day_vote is day_vote_nodes.day_vote
    assert day._broadcast_vote_details is day_vote_nodes._broadcast_vote_details
    assert day.resolve_vote is day_vote_nodes.resolve_vote
    assert day.resolve_exile is day_vote_nodes.resolve_exile


def test_day_facade_monkeypatch_propagates_vote_dependencies(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import day

    day_vote_nodes = importlib.import_module("werewolf_agent.runtime.nodes.day_vote")

    def fake_jb(*args, **kwargs):
        return args[3], None

    def fake_agent_vote(*args, **kwargs):
        return None

    monkeypatch.setattr(day, "_jb", fake_jb)
    monkeypatch.setattr(day, "agent_day_vote", fake_agent_vote)

    assert day_vote_nodes._jb is fake_jb
    assert day_vote_nodes.agent_day_vote is fake_agent_vote
