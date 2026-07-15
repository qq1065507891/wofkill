# -*- coding: utf-8 -*-
"""
验证暂停、启动与 GameEvent V1/V2 replay 的事件溯源行为。

作者: Project contributors
修改日期: 2026-07-15
"""

from __future__ import annotations

from dataclasses import replace


from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.graph import _new_engine
from werewolf_agent.runtime.replay import replay_from_events

class TestPauseResumeEventSourcing:
    """Pause/resume must use event sourcing (GameEvent + reducer),
    not direct object.__setattr__ mutation."""

    def test_pause_creates_event_via_replace(self) -> None:
        """Pausing a game creates a game_paused event and sets paused=True via replace."""
        gs = GameState(game_id="pause_evt", phase="night")
        event = GameEvent(type="game_paused", payload={
            "game_id": "pause_evt", "phase": "night",
        })
        gs = replace(gs, paused=True, events=gs.events + [event])
        assert gs.paused is True
        assert any(e.type == "game_paused" for e in gs.events)

    def test_resume_creates_event_via_replace(self) -> None:
        """Resuming a game creates a game_resumed event and sets paused=False via replace."""
        gs = GameState(game_id="resume_evt", phase="night", paused=True)
        event = GameEvent(type="game_resumed", payload={
            "game_id": "resume_evt", "phase": "night",
        })
        gs = replace(gs, paused=False, events=gs.events + [event])
        assert gs.paused is False
        assert any(e.type == "game_resumed" for e in gs.events)

    def test_pause_event_reduces_correctly(self) -> None:
        """game_paused event can be replayed through RuleEngine reducer."""
        engine = _new_engine()
        gs = GameState(game_id="pause_reduce", phase="night")
        event = GameEvent(type="game_paused", payload={"game_id": "pause_reduce"})
        gs = engine.reduce_event(gs, event)
        assert gs.paused is True
        assert any(e.type == "game_paused" for e in gs.events)

    def test_resume_event_reduces_correctly(self) -> None:
        """game_resumed event can be replayed through RuleEngine reducer."""
        engine = _new_engine()
        gs = GameState(game_id="resume_reduce", phase="night", paused=True)
        event = GameEvent(type="game_resumed", payload={"game_id": "resume_reduce"})
        gs = engine.reduce_event(gs, event)
        assert gs.paused is False

    def test_pause_resume_full_replay(self) -> None:
        """Pause then resume events replay correctly from event log."""
        engine = _new_engine()
        gs = GameState(game_id="pause_replay", phase="night")
        events = [
            GameEvent(type="game_paused", payload={"game_id": "pause_replay"}),
            GameEvent(type="game_resumed", payload={"game_id": "pause_replay"}),
        ]
        replayed = engine.reduce_events(gs, events)
        assert replayed.paused is False
        assert len(replayed.events) == 2

    def test_api_pause_uses_event_sourcing(self) -> None:
        """FastAPI pause endpoint creates game_paused event, not direct mutation."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        # Create and start game
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", "caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        # Pause
        r = client.post(f"/games/{game_id}/pause", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        # Check that game_paused event exists in the game's events
        r = client.get(f"/games/{game_id}/timeline", params={
            "caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full",
        })
        assert r.status_code == 200
        event_types = [e["event_type"] for e in r.json().get("events", [])]
        assert "game_paused" in event_types

    def test_api_resume_uses_event_sourcing(self) -> None:
        """FastAPI resume endpoint creates game_resumed event."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", "caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        r = client.post(f"/games/{game_id}/pause", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        r = client.post(f"/games/{game_id}/resume", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        r = client.get(f"/games/{game_id}/timeline", params={
            "caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full",
        })
        assert r.status_code == 200
        event_types = [e["event_type"] for e in r.json().get("events", [])]
        assert "game_resumed" in event_types

class TestStartGameEventSourcing:
    """start_game endpoint uses replace() + GameEvent, not object.__setattr__."""

    def test_no_object_setattr_in_app(self) -> None:
        """Verify no object.__setattr__ remains in app.py."""
        from pathlib import Path
        app_src = Path(__file__).parent.parent.parent / "werewolf_agent" / "api" / "app.py"
        content = app_src.read_text(encoding="utf-8")
        assert "object.__setattr__" not in content, \
            "object.__setattr__ still exists in app.py"

    def test_game_started_event_in_state(self) -> None:
        """RuleEngine reducer handles game_started event."""
        engine = _new_engine()
        gs = GameState(game_id="start_test", ruleset_id="pre_witch_hunter_idiot_mixed", phase="setup")
        players_data = {}
        for i, role in enumerate(["werewolf"] * 4 + ["villager"] * 3 + ["seer", "witch", "hunter", "idiot", "hybrid"]):
            pid = f"p{i + 1:02d}"
            players_data[pid] = {"id": pid, "role": role}
        event = GameEvent(type="game_started", payload={"game_id": "start_test", "players": players_data})
        result = engine.reduce_event(gs, event)
        assert result.phase == "night"
        assert len(result.players) == 12
        assert result.players["p01"].role == "werewolf"
        assert result.players["p05"].role == "villager"
        assert result.players["p08"].role == "seer"
        started_events = [e for e in result.events if e.type == "game_started"]
        assert len(started_events) == 1

    def test_game_started_replay(self) -> None:
        """game_started event is replayable through reducer."""
        engine = _new_engine()
        gs = GameState(game_id="replay_start", ruleset_id="pre_witch_hunter_idiot_mixed", phase="setup")
        players_data = {}
        for i, role in enumerate(["werewolf"] * 4 + ["villager"] * 3 + ["seer", "witch", "hunter", "idiot", "hybrid"]):
            pid = f"p{i + 1:02d}"
            players_data[pid] = {"id": pid, "role": role}
        event = GameEvent(type="game_started", payload={"game_id": "replay_start", "players": players_data})
        result = engine.reduce_event(gs, event)
        # Replay from scratch
        replayed = engine.reduce_event(gs, event)
        assert replayed.phase == result.phase
        assert replayed.players.keys() == result.players.keys()
        for pid in result.players:
            assert replayed.players[pid].role == result.players[pid].role

    def test_api_start_game_creates_event(self) -> None:
        """FastAPI start_game endpoint creates game_started event."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", "caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        # Check event in timeline
        r = client.get(f"/games/{game_id}/timeline", params={
            "caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full",
        })
        assert r.status_code == 200
        event_types = [e["event_type"] for e in r.json().get("events", [])]
        assert "game_started" in event_types

    def test_api_start_game_players_set(self) -> None:
        """After start_game, public state has 12 players and phase is night."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", "caller_id": "mod1", "caller_role": "moderator"})
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        r = client.get(f"/games/{game_id}/public-state")
        assert r.status_code == 200
        data = r.json()
        assert data.get("phase") == "night"
        assert len(data.get("players", {})) == 12

    def test_game_started_reducer_idempotent(self) -> None:
        """Applying game_started twice doesn't corrupt state."""
        engine = _new_engine()
        gs = GameState(game_id="idem_test", ruleset_id="pre_witch_hunter_idiot_mixed", phase="setup")
        players_data = {}
        for i, role in enumerate(["werewolf"] * 4 + ["villager"] * 3 + ["seer", "witch", "hunter", "idiot", "hybrid"]):
            pid = f"p{i + 1:02d}"
            players_data[pid] = {"id": pid, "role": role}
        event = GameEvent(type="game_started", payload={"game_id": "idem_test", "players": players_data})
        result1 = engine.reduce_event(gs, event)
        result2 = engine.reduce_event(result1, event)
        assert result2.phase == "night"
        assert len(result2.players) == 12
        assert len([e for e in result2.events if e.type == "game_started"]) == 2
def test_replay_orders_v2_events_by_sequence_number() -> None:
    class CapturingEngine:
        def reduce_events(self, state, events):
            return replace(state, events=list(events))

    events = [
        GameEvent(type="second", sequence_number=1),
        GameEvent(type="first", sequence_number=0),
    ]

    replayed = replay_from_events(CapturingEngine(), GameState(), events)

    assert [event.type for event in replayed.events] == ["first", "second"]


def test_replay_preserves_v1_array_order_without_sequence_numbers() -> None:
    class CapturingEngine:
        def reduce_events(self, state, events):
            return replace(state, events=list(events))

    events = [GameEvent(type="second"), GameEvent(type="first")]

    replayed = replay_from_events(CapturingEngine(), GameState(), events)

    assert replayed.events == events


def test_replay_preserves_legacy_prefix_before_v2_suffix() -> None:
    class CapturingEngine:
        def reduce_events(self, state, events):
            return replace(state, events=list(events))

    events = [
        GameEvent(type="legacy_prefix"),
        GameEvent(type="v2_suffix", sequence_number=1),
    ]

    replayed = replay_from_events(CapturingEngine(), GameState(), events)

    assert replayed.events == events


def test_replay_preserves_mixed_log_positions_without_trusting_v2_sequence() -> None:
    class CapturingEngine:
        def reduce_events(self, state, events):
            return replace(state, events=list(events))

    events = [
        GameEvent(type="v2_before", sequence_number=7),
        GameEvent(type="legacy_middle"),
        GameEvent(type="v2_after", sequence_number=2),
    ]

    replayed = replay_from_events(CapturingEngine(), GameState(), events)

    assert replayed.events == events
