# -*- coding: utf-8 -*-
"""
验证 player latency helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_player_latency.py -q
"""

from __future__ import annotations

from types import SimpleNamespace


def test_player_latency_helper_remains_compatibly_importable() -> None:
    from werewolf_agent.agents import player
    from werewolf_agent.agents import player_latency

    assert player._latency_from_result is player_latency.latency_from_result


def test_latency_from_result_returns_zero_without_usage() -> None:
    from werewolf_agent.agents.player_latency import latency_from_result

    assert latency_from_result(SimpleNamespace(usage=None)) == 0
