# -*- coding: utf-8 -*-
"""
验证夜晚身份行动节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_night_specialists.py -q
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine


class _Registry:
    def get_agent(self, _player_id: str) -> None:
        return None


def test_night_specialist_nodes_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import night
    from werewolf_agent.runtime.nodes import night_specialists

    assert night.night_witch is night_specialists.night_witch
    assert night.night_seer is night_specialists.night_seer
    assert night.first_night_hybrid_master is night_specialists.first_night_hybrid_master


def test_specialist_nodes_respect_old_agent_monkeypatch_path(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import night as night_mod

    calls: list[Any] = []

    def fake_dispatch(_state: dict[str, Any], fn: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append(fn)
        if fn is fake_witch:
            return {"use_antidote": False, "poison_target_id": None}
        if fn is fake_seer:
            return {"seer_target_id": "p03"}
        if fn is fake_hybrid:
            return {"master_target_id": "p03"}
        return {}

    def fake_witch() -> None:
        return None

    def fake_seer() -> None:
        return None

    def fake_hybrid() -> None:
        return None

    players = {
        "p01": PlayerState(id="p01", role="witch", alive=True),
        "p02": PlayerState(id="p02", role="seer", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
        "p04": PlayerState(id="p04", role="hybrid", alive=True),
    }
    gs = GameState(game_id="night_agent_patch", players=players, phase="night", night_number=1)
    state = {
        "game_state": gs,
        "engine": RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml"),
        "agent_registry": _Registry(),
    }

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch)
    monkeypatch.setattr(night_mod, "agent_night_witch", fake_witch)
    monkeypatch.setattr(night_mod, "agent_night_seer", fake_seer)
    monkeypatch.setattr(night_mod, "agent_hybrid_choose_master", fake_hybrid)

    witch_result = night_mod.night_witch(state)
    seer_result = night_mod.night_seer({**state, "game_state": witch_result["game_state"]})
    hybrid_state = {**state, "game_state": replace(seer_result["game_state"], hybrid_master_id=None)}
    night_mod.first_night_hybrid_master(hybrid_state)

    assert fake_witch in calls
    assert fake_seer in calls
    assert fake_hybrid in calls
