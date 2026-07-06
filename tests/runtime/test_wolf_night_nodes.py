# -*- coding: utf-8 -*-
"""
验证狼人夜晚节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_wolf_night_nodes.py -q
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState, PlayerState


class _Registry:
    def get_agent(self, _player_id: str) -> None:
        return None


def test_wolf_night_nodes_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import night
    from werewolf_agent.runtime.nodes import wolf_night_nodes

    assert night._legacy_wolf_consensus is wolf_night_nodes._legacy_wolf_consensus
    assert night.wolf_discussion is wolf_night_nodes.wolf_discussion
    assert night._build_fallback_wolf_team_plan is wolf_night_nodes._build_fallback_wolf_team_plan
    assert night.wolf_team_plan_node is wolf_night_nodes.wolf_team_plan_node
    assert night.wolf_consensus is wolf_night_nodes.wolf_consensus


def test_wolf_nodes_respect_old_agent_monkeypatch_path(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import night as night_mod

    calls: list[Any] = []

    def fake_dispatch(_state: dict[str, Any], fn: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append(fn)
        if fn is fake_wolf_discussion:
            return {"speech_text": "今晚刀 p03"}
        if fn is fake_wolf_consensus:
            return {"wolf_action": "kill", "wolf_kill_target_id": "p03"}
        return {}

    def fake_wolf_discussion() -> None:
        return None

    def fake_wolf_consensus() -> None:
        return None

    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    gs = GameState(game_id="wolf_agent_patch", players=players, phase="night", night_number=1)
    state = {"game_state": gs, "agent_registry": _Registry()}

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch)
    monkeypatch.setattr(night_mod, "agent_wolf_discussion", fake_wolf_discussion)
    monkeypatch.setattr(night_mod, "agent_wolf_consensus", fake_wolf_consensus)

    discussion_result = night_mod.wolf_discussion(state)
    night_mod.wolf_consensus({**state, "game_state": discussion_result["game_state"]})

    assert fake_wolf_discussion in calls
    assert fake_wolf_consensus in calls
