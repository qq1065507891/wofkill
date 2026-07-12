# -*- coding: utf-8 -*-
"""
验证 day death / last-words 节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> python -m pytest tests/runtime/test_day_deaths.py -q
"""

from __future__ import annotations

import importlib
from dataclasses import replace

from werewolf_agent.core.models import GameEvent, GameState, PlayerState

from werewolf_agent.runtime.graph import _new_engine, resolve_exile, route_after_post_exile


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


def _exile_state(*, wolves: int, good: int, exiled_role: str = "villager"):
    engine = _new_engine()
    players = {
        **{f"w{i}": PlayerState(id=f"w{i}", role="werewolf") for i in range(wolves)},
        **{f"g{i}": PlayerState(id=f"g{i}", role="villager") for i in range(good)},
        "exiled": PlayerState(id="exiled", role=exiled_role),
    }
    gs = GameState(
        game_id="terminal_exile",
        players=players,
        phase="day",
        day_number=1,
        events=[GameEvent(type="vote_resolved", payload={"exiled": "exiled"})],
    )
    return {"game_state": gs, "engine": engine}


def test_exile_parity_routes_to_victory_before_last_words(monkeypatch) -> None:
    state = _exile_state(wolves=3, good=1)
    result = resolve_exile(state)
    calls = 0

    def forbidden_last_words(_state):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.day_deaths.exile_last_words",
        forbidden_last_words,
    )

    assert result["game_state"].winning_faction == "werewolf"
    assert route_after_post_exile({**state, **result}) == "reflection"
    assert calls == 0


def test_last_wolf_exile_skips_last_words() -> None:
    state = _exile_state(wolves=0, good=2, exiled_role="werewolf")
    state["game_state"] = replace(
        state["game_state"],
        players={
            **state["game_state"].players,
            "exiled": PlayerState(id="exiled", role="werewolf"),
        },
    )
    result = resolve_exile(state)

    assert result["game_state"].winning_faction == "good"
    assert route_after_post_exile({**state, **result}) == "reflection"
