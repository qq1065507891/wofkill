# -*- coding: utf-8 -*-
"""
验证 day death / last-words 节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_day_deaths.py -q
"""

from __future__ import annotations

import importlib


def test_day_death_nodes_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import day
    from werewolf_agent.runtime.nodes import day_deaths

    assert day.announce_deaths is day_deaths.announce_deaths
    assert day.announce_deaths_with_badge_loss is day_deaths.announce_deaths_with_badge_loss
    assert day.night_death_last_words is day_deaths.night_death_last_words
    assert day.exile_last_words is day_deaths.exile_last_words
    assert day._death_reason_label is day_deaths._death_reason_label


def test_day_facade_monkeypatch_propagates_last_words_dependencies(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import day

    day_deaths = importlib.import_module("werewolf_agent.runtime.nodes.day_deaths")

    def fake_dispatch(*args, **kwargs):
        return None

    def fake_last_words(*args, **kwargs):
        return None

    monkeypatch.setattr(day, "_dispatch_agent", fake_dispatch)
    monkeypatch.setattr(day, "agent_exile_last_words", fake_last_words)

    assert day_deaths._dispatch_agent is fake_dispatch
    assert day_deaths.agent_exile_last_words is fake_last_words
