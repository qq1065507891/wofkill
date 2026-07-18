# -*- coding: utf-8 -*-
"""
验证所有生产狼人落刀路径都生成稳定、可回放且互不冲突的语义 trace。

作者: Project contributors
创建日期: 2026-07-18

使用示例:
    >>> python -m pytest tests/runtime/test_wolf_kill_selection_trace.py -q
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pytest

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.event_metadata import (
    deserialize_game_event,
    new_game_event,
    serialize_game_event,
)
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_validated_wolf_target_stance,
)


SelectedEventFactory = Callable[[], GameEvent]


class _AgentRegistryStub:
    """仅满足 legacy 日志展示所需的最小 registry 协议。"""

    def get_agent(self, _player_id: str) -> None:
        return None


def _players(*, single_wolf: bool = False) -> dict[str, PlayerState]:
    """构造覆盖主刀、备刀和恢复路径所需的最小玩家集合。"""
    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="seer"),
        "v3": PlayerState(id="v3", role="witch"),
    }
    if not single_wolf:
        players.update({
            "w2": PlayerState(id="w2", role="werewolf"),
            "w3": PlayerState(id="w3", role="werewolf"),
        })
    return players


def _selected_event(game_state: GameState) -> GameEvent:
    """读取一次生产决策追加的落刀事件。"""
    return next(
        event
        for event in reversed(game_state.events)
        if event.type == "wolf_kill_selected"
    )


def _append_stance(
    game_state: GameState,
    *,
    wolf_id: str,
    target_id: str,
    priority: str = "primary",
) -> GameState:
    """通过生产校验器追加可授权落刀的结构化立场。"""
    round_number = len(game_state.events) + 1
    payload = {
        "wolf_id": wolf_id,
        "round": round_number,
        "night_number": game_state.night_number,
        "text": "",
    }
    source = new_game_event(
        game_state,
        "wolf_discussion",
        payload,
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )
    stance = build_validated_wolf_target_stance(
        game_state,
        source,
        wolf_id=wolf_id,
        round_number=round_number,
        raw_stance={
            "target_id": target_id,
            "stance": "support",
            "priority": priority,
        },
    )
    event = replace(
        source,
        payload={**payload, "target_stance": stance.model_dump()},
    )
    return replace(game_state, events=[*game_state.events, event])


def _explicit_state_selected(game_id: str) -> GameEvent:
    from werewolf_agent.runtime.nodes.wolf_consensus import (
        _legacy_wolf_consensus,
    )

    game_state = GameState(
        game_id=game_id,
        players=_players(),
        phase="night",
        night_number=3,
    )
    result = _legacy_wolf_consensus({
        "game_state": game_state,
        "wolf_action": "kill",
        "wolf_kill_target_id": "v1",
    })
    return _selected_event(result["game_state"])


def _legacy_agent_selected(
    monkeypatch: pytest.MonkeyPatch,
    game_id: str,
) -> GameEvent:
    from werewolf_agent.runtime.nodes import night as night_module
    from werewolf_agent.runtime.nodes.wolf_consensus import (
        _legacy_wolf_consensus,
    )

    game_state = GameState(
        game_id=game_id,
        players=_players(),
        phase="night",
        night_number=3,
    )
    monkeypatch.setattr(
        night_module,
        "_dispatch_agent",
        lambda *_args, **_kwargs: {
            "wolf_action": "kill",
            "wolf_kill_target_id": "v1",
            "action_traces": {},
        },
    )
    result = _legacy_wolf_consensus({
        "game_state": game_state,
        "agent_registry": _AgentRegistryStub(),
    })
    return _selected_event(result["game_state"])


def _structured_selected(
    *,
    game_id: str,
    single_wolf: bool,
    use_backup: bool,
) -> GameEvent:
    from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill

    game_state = GameState(
        game_id=game_id,
        players=_players(single_wolf=single_wolf),
        phase="night",
        night_number=3,
    )
    supporting_wolves = ("w1",) if single_wolf else ("w1", "w2")
    for wolf_id in supporting_wolves:
        game_state = _append_stance(
            game_state,
            wolf_id=wolf_id,
            target_id="v1",
        )
        if use_backup:
            game_state = _append_stance(
                game_state,
                wolf_id=wolf_id,
                target_id="v2",
                priority="backup",
            )
    if use_backup:
        players = dict(game_state.players)
        players["v1"] = replace(players["v1"], alive=False)
        game_state = replace(game_state, players=players)

    result = _planned_wolf_kill({"game_state": game_state})

    assert result is not None
    return _selected_event(result["game_state"])


def _forced_fallback_selected(game_id: str) -> GameEvent:
    from werewolf_agent.runtime.nodes.node_helpers import _force_wolf_kill

    game_state = GameState(
        game_id=game_id,
        players=_players(),
        phase="night",
        night_number=3,
    )
    result = _force_wolf_kill(game_state, "legacy-provider-fallback")
    return _selected_event(result["game_state"])


def _forced_recovery_selected(game_id: str) -> GameEvent:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    prior_events = [
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={
                "night_number": 1,
                "reason": "provider_unavailable",
                "no_kill_decision": {
                    "reason_code": "provider_unavailable",
                    "consecutive_pre_resolution_no_kill_count": 1,
                    "forced_recovery_applied": False,
                    "recovered_target_id": None,
                },
            },
        ),
        GameEvent(
            type="wolf_no_kill_declared",
            payload={
                "night_number": 2,
                "reason": "strategic_abstain",
                "no_kill_decision": {
                    "reason_code": "strategic_abstain",
                    "consecutive_pre_resolution_no_kill_count": 2,
                    "forced_recovery_applied": False,
                    "recovered_target_id": None,
                },
            },
        ),
    ]
    game_state = GameState(
        game_id=game_id,
        players=_players(),
        events=prior_events,
        phase="night",
        night_number=3,
    )
    result = NoKillPolicy().resolve(
        game_state,
        reason_code="true_tie",
        primary_positive_support={"v1": 1},
    )
    return _selected_event(result["game_state"])


def _route_factories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    game_id: str = "trace-wolf-selection",
) -> dict[str, SelectedEventFactory]:
    """集中列出每条真实生产落刀路线，防止后续新增写点漏测。"""
    return {
        "legacy_agent": lambda: _legacy_agent_selected(monkeypatch, game_id),
        "explicit_state": lambda: _explicit_state_selected(game_id),
        "stance_majority_primary": lambda: _structured_selected(
            game_id=game_id,
            single_wolf=False,
            use_backup=False,
        ),
        "stance_single_wolf_primary": lambda: _structured_selected(
            game_id=game_id,
            single_wolf=True,
            use_backup=False,
        ),
        "stance_backup": lambda: _structured_selected(
            game_id=game_id,
            single_wolf=False,
            use_backup=True,
        ),
        "forced_fallback": lambda: _forced_fallback_selected(game_id),
        "forced_recovery": lambda: _forced_recovery_selected(game_id),
    }


def test_every_production_kill_route_has_stable_replayable_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一路线重复执行和事件序列化往返都必须保留同一 trace。"""
    for route_name, factory in _route_factories(monkeypatch).items():
        first = factory()
        second = factory()

        assert first.trace_id, route_name
        assert second.trace_id == first.trace_id, route_name
        restored = deserialize_game_event(serialize_game_event(first))
        assert restored.trace_id == first.trace_id, route_name


def test_semantically_distinct_kill_routes_do_not_share_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一夜可能出现的不同决策链必须拥有不同语义身份。"""
    trace_by_route = {
        route_name: factory().trace_id
        for route_name, factory in _route_factories(monkeypatch).items()
    }

    assert all(trace_by_route.values())
    assert len(set(trace_by_route.values())) == len(trace_by_route), trace_by_route


def test_v1_wolf_kill_event_does_not_gain_trace_during_deserialization() -> None:
    """旧事件读取保持兼容，不得猜测或回填新的语义 trace。"""
    restored = deserialize_game_event({
        "type": "wolf_kill_selected",
        "payload": {"night_number": 1, "target_id": "v1"},
    })

    assert restored.schema_version is None
    assert restored.trace_id is None
