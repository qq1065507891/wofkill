# -*- coding: utf-8 -*-
"""
警长节点拆分后的兼容 facade 测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/runtime/test_sheriff_node_split.py
"""

from __future__ import annotations

import importlib
from dataclasses import replace

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.graph import _new_engine
from werewolf_agent.runtime.nodes import sheriff as sheriff_facade


sheriff_endorse = importlib.import_module("werewolf_agent.runtime.nodes.sheriff_endorse")
sheriff_registration = importlib.import_module("werewolf_agent.runtime.nodes.sheriff_registration")
sheriff_speech = importlib.import_module("werewolf_agent.runtime.nodes.sheriff_speech")
sheriff_vote = importlib.import_module("werewolf_agent.runtime.nodes.sheriff_vote")


def test_sheriff_symbols_are_reexported_from_facade_modules() -> None:
    assert sheriff_facade.sheriff_first_day_entry is sheriff_registration.sheriff_first_day_entry
    assert sheriff_facade.sheriff_registration is sheriff_registration.sheriff_registration
    assert sheriff_facade.sheriff_withdraw is sheriff_registration.sheriff_withdraw
    assert sheriff_facade.sheriff_vote is sheriff_vote.sheriff_vote
    assert sheriff_facade.sheriff_speech is sheriff_speech.sheriff_speech
    assert sheriff_facade.sheriff_endorse is sheriff_endorse.sheriff_endorse
    assert sheriff_facade._sheriff_endorse_adapter is sheriff_endorse._sheriff_endorse_adapter


def test_sheriff_facade_dispatch_patch_reaches_speech_module(monkeypatch) -> None:
    calls: list[str] = []

    def fake_dispatch_agent(_state, _fn, candidate_id, *_args, **_kwargs):
        calls.append(candidate_id)
        return {"speech_text": f"{candidate_id} 发言", "action_trace": {}}

    class Registry:
        def get_agent(self, _player_id):
            return object()

    gs = GameState(
        game_id="sheriff_split_patch",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
        sheriff_candidates=["p01", "p02"],
    )

    monkeypatch.setattr(sheriff_facade, "_dispatch_agent", fake_dispatch_agent)

    result = sheriff_facade.sheriff_speech({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
    })

    speakers = {
        event.payload.get("speaker")
        for event in result["game_state"].events
        if event.type == "sheriff_speech"
    }
    assert set(calls) == {"p01", "p02"}
    assert speakers == {"p01", "p02"}


def test_sheriff_withdraw_scripted_fallback_still_uses_engine() -> None:
    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = replace(
        GameState(game_id="sheriff_withdraw_split", day_number=1, players=players),
        sheriff_candidates=["p01", "p02"],
    )

    result = sheriff_facade.sheriff_withdraw({
        "game_state": gs,
        "engine": _new_engine(),
        "sheriff_withdrawing": ["p02"],
    })

    assert result["sheriff_candidates"] == ["p01"]
    assert result["sheriff_withdrawing"] == ["p02"]
