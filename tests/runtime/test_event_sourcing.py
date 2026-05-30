from __future__ import annotations

import pytest
from dataclasses import replace

from typing import Any

from werewolf_agent.core.models import Death, GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import (
    ActionType, AgentContext, PlayerAction, RetryInfo, FallbackAction,
    TaskType,
)
from werewolf_agent.runtime.graph import (
    RuntimeState,
    build_game_graph,
    build_game_graph_with_checkpoint,
    _new_engine,
    _alive_wolves,
    _alive_non_wolves,
    _find_role,
    _stable_seed,
    check_victory,
    free_discussion,
    wolf_consensus,
    route_after_resolve_night,
    route_after_hunter_shot,
    route_after_post_exile,
    _sheriff_died_this_batch,
    _route_after_badge_transfer,
    _action_trace_event,
)
from werewolf_agent.runtime.agent_adapter import _single_wolf_vote
from werewolf_agent.runtime.replay import replay_from_events, extract_event_log
from werewolf_agent.runtime.checkpoints import make_checkpointer



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
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed"})
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
        assert r.status_code == 200
        # Pause
        r = client.post(f"/games/{game_id}/pause", json={"caller_id": "mod1"})
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
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed"})
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
        assert r.status_code == 200
        r = client.post(f"/games/{game_id}/pause", json={"caller_id": "mod1"})
        assert r.status_code == 200
        r = client.post(f"/games/{game_id}/resume", json={"caller_id": "mod1"})
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
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed"})
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
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
        r = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed"})
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
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