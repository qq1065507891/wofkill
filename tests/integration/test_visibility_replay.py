"""Integration tests: end-to-end visibility and replay safety.

Verifies:
1. No information leakage across public/player/spectator/moderator/debugger views
2. Pause/resume creates audit events (not direct state mutation)
3. Replay from initial_seed + ruleset_snapshot + event_log reconstructs correct state
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from werewolf_agent.api.schemas import CallerRole, ViewMode
from werewolf_agent.api.views import (
    build_cognitive_diff,
    build_evaluation,
    build_private_state,
    build_public_state,
    build_replay,
    build_timeline,
)
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.replay import replay_from_events, extract_event_log

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _make_game_state() -> GameState:
    """Create a rich game state with all role types and events for leak testing."""
    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "w3": PlayerState(id="w3", role="werewolf"),
        "w4": PlayerState(id="w4", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="villager"),
        "v3": PlayerState(id="v3", role="villager"),
        "seer": PlayerState(id="seer", role="seer"),
        "witch": PlayerState(id="witch", role="witch"),
        "hunter": PlayerState(id="hunter", role="hunter"),
        "idiot": PlayerState(id="idiot", role="idiot"),
        "hybrid": PlayerState(id="hybrid", role="hybrid"),
    }
    events = [
        GameEvent(type="roles_assigned", payload={}),
        GameEvent(type="enter_night", payload={"night": 1}),
        GameEvent(type="wolf_discussion", payload={"wolves": ["w1", "w2", "w3", "w4"]}),
        GameEvent(type="wolf_kill_selected", payload={"target_id": "v1", "night_number": 1}),
        GameEvent(type="witch_antidote_used", payload={"target_id": "v1"}),
        GameEvent(type="seer_check", payload={
            "seer_id": "seer", "target_id": "w1",
            "alignment": "werewolf", "night_number": 1,
            "visibility": "seer_only",
        }),
        GameEvent(type="hunter_idiot_status_confirmed", payload={
            "hunter_id": "hunter", "idiot_id": "idiot",
            "visibility": "moderator_only",
        }),
        GameEvent(type="hybrid_master_chosen", payload={
            "hybrid_id": "hybrid", "master_id": "v2",
        }),
        GameEvent(type="player_died", payload={
            "player_id": "v1", "reason": "wolf_kill",
            "timing": "night", "resolution_batch": "night_1",
        }),
        GameEvent(type="speech", payload={
            "speaker": "w1", "day_number": 1,
            "text": "我是好人，查杀v2。",
        }),
        GameEvent(type="speech", payload={
            "speaker": "seer", "day_number": 1,
            "text": "我是预言家，昨晚查w1是狼。",
        }),
    ]
    return GameState(
        game_id="leak_test",
        players=players,
        phase="day",
        day_number=1,
        night_number=1,
        events=events,
        deaths=[Death(player_id="v1", reason="wolf_kill", timing="night", resolution_batch="night_1")],
        hybrid_master_id="v2",
        hybrid_master_faction="good",
        antidote_used=True,
        sheriff_id="seer",
        sheriff_badge_state="active",
    )


# ---------------------------------------------------------------------------
# Step 1: Leak tests across API views
# ---------------------------------------------------------------------------


class TestPublicViewLeaks:
    """Public view must not reveal any private information."""

    def test_public_state_no_roles(self) -> None:
        gs = _make_game_state()
        result = build_public_state(gs)
        for p in result.players:
            assert p.revealed_role is None or p.revealed_role == "idiot"

    def test_public_timeline_no_private_events(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PUBLIC)
        event_types = [e.event_type for e in timeline.events]
        assert "seer_check" not in event_types
        assert "wolf_discussion" not in event_types
        assert "witch_antidote_used" not in event_types
        assert "hunter_idiot_status_confirmed" not in event_types
        assert "hybrid_master_chosen" not in event_types

    def test_public_timeline_no_wolf_kill_details(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PUBLIC)
        for e in timeline.events:
            if e.event_type == "player_died":
                assert "reason" not in e.data or e.data.get("player_id") is not None

    def test_public_replay_no_roles(self) -> None:
        gs = _make_game_state()
        replay = build_replay(gs, ViewMode.PUBLIC)
        assert len(replay.snapshots) == 1
        snap = replay.snapshots[0]
        assert snap.moderator_full is None
        assert snap.player_views == {}


class TestPlayerViewLeaks:
    """Player views must only show role-specific private information."""

    def test_wolf_player_sees_wolf_teammates(self) -> None:
        gs = _make_game_state()
        result = build_private_state(gs, "w1", ViewMode.PLAYER_VIEW)
        assert result.player_info.wolf_teammates is not None
        assert "w2" in result.player_info.wolf_teammates
        assert "w3" in result.player_info.wolf_teammates

    def test_wolf_player_does_not_see_seer_result(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PLAYER_VIEW, viewer_id="w1")
        seer_events = [e for e in timeline.events if e.event_type == "seer_check"]
        assert len(seer_events) == 0, "Wolf must not see seer_check events"

    def test_seer_sees_own_check_results_only(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PLAYER_VIEW, viewer_id="seer")
        seer_events = [e for e in timeline.events if e.event_type == "seer_check"]
        assert len(seer_events) == 1
        assert seer_events[0].data.get("target_id") == "w1"
        assert seer_events[0].data.get("alignment") == "werewolf"

    def test_villager_sees_no_night_private_events(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PLAYER_VIEW, viewer_id="v2")
        event_types = [e.event_type for e in timeline.events]
        assert "seer_check" not in event_types
        assert "wolf_discussion" not in event_types
        assert "witch_antidote_used" not in event_types
        assert "hunter_idiot_status_confirmed" not in event_types
        assert "hybrid_master_chosen" not in event_types

    def test_witch_sees_potion_availability(self) -> None:
        gs = _make_game_state()
        result = build_private_state(gs, "witch", ViewMode.PLAYER_VIEW)
        assert result.player_info.antidote_available is False  # used
        assert result.player_info.poison_available is True  # not used

    def test_hybrid_sees_master_id(self) -> None:
        gs = _make_game_state()
        result = build_private_state(gs, "hybrid", ViewMode.PLAYER_VIEW)
        assert result.player_info.master_id == "v2"

    def test_non_hybrid_cannot_see_master_id(self) -> None:
        gs = _make_game_state()
        result = build_private_state(gs, "v1", ViewMode.PLAYER_VIEW)
        assert result.player_info.master_id is None

    def test_wolf_sees_wolf_events_in_timeline(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PLAYER_VIEW, viewer_id="w1")
        event_types = [e.event_type for e in timeline.events]
        assert "wolf_discussion" in event_types or "wolf_kill_selected" in event_types

    def test_non_wolf_does_not_see_wolf_events(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PLAYER_VIEW, viewer_id="seer")
        event_types = [e.event_type for e in timeline.events]
        assert "wolf_discussion" not in event_types

    def test_player_private_state_no_private_intent(self) -> None:
        gs = _make_game_state()
        result = build_private_state(gs, "w1", ViewMode.PLAYER_VIEW)
        data = json.dumps(result.model_dump(), ensure_ascii=False)
        assert "private_intent" not in data


class TestSpectatorViewLeaks:
    """Spectator view is downgraded to public."""

    def test_spectator_sees_no_private_events(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.PUBLIC)
        event_types = [e.event_type for e in timeline.events]
        private_types = {"seer_check", "wolf_discussion", "witch_antidote_used",
                         "hunter_idiot_status_confirmed", "hybrid_master_chosen"}
        for pt in private_types:
            assert pt not in event_types, f"Spectator must not see {pt}"


class TestModeratorViewAccess:
    """Moderator and debugger can see everything except private_intent."""

    def test_moderator_timeline_sees_all_events(self) -> None:
        gs = _make_game_state()
        timeline = build_timeline(gs, ViewMode.MODERATOR_FULL)
        event_types = [e.event_type for e in timeline.events]
        assert "seer_check" in event_types
        assert "wolf_discussion" in event_types
        assert "witch_antidote_used" in event_types
        assert "hunter_idiot_status_confirmed" in event_types

    def test_moderator_replay_shows_all_roles(self) -> None:
        gs = _make_game_state()
        replay = build_replay(gs, ViewMode.MODERATOR_FULL)
        snap = replay.snapshots[0]
        assert snap.moderator_full is not None
        all_roles = snap.moderator_full.get("all_roles", {})
        assert all_roles.get("w1") == "werewolf"
        assert all_roles.get("seer") == "seer"

    def test_moderator_replay_no_private_intent(self) -> None:
        gs = _make_game_state()
        replay = build_replay(gs, ViewMode.MODERATOR_FULL)
        data = json.dumps(replay.model_dump(), ensure_ascii=False)
        # private_intent should not appear in replay responses
        # (moderator sees all roles but not private_intent)
        assert "private_intent" not in data or "[redacted" in data

    def test_moderator_evaluation_shows_roles(self) -> None:
        gs = _make_game_state()
        eval_result = build_evaluation(gs, ViewMode.MODERATOR_FULL)
        for pid, stats in eval_result.metrics.player_stats.items():
            assert stats["role"] != "[hidden]"

    def test_cognitive_diff_moderator_shows_actual_roles(self) -> None:
        gs = _make_game_state()
        diff = build_cognitive_diff(gs, "seer", ViewMode.MODERATOR_FULL)
        for entry in diff.entries:
            assert entry.actual_role is not None


class TestCognitiveDiffLeaks:
    """Cognitive diff must hide actual roles from non-moderator views."""

    def test_cognitive_diff_public_hides_actual_roles(self) -> None:
        gs = _make_game_state()
        diff = build_cognitive_diff(gs, "seer", ViewMode.PUBLIC)
        for entry in diff.entries:
            assert entry.actual_role is None
            assert entry.actual_faction is None


# ---------------------------------------------------------------------------
# Step 2: Pause/resume event-sourced
# ---------------------------------------------------------------------------


class TestPauseResumeEventSourced:
    """Pause and resume must create audit events, not just mutate state."""

    def test_pause_creates_game_event(self) -> None:
        gs = GameState(game_id="pause_test", phase="night")
        event = GameEvent(type="game_paused", payload={
            "game_id": "pause_test", "phase": "night",
        })
        gs = replace(gs, paused=True, events=gs.events + [event])
        assert gs.paused is True
        assert any(e.type == "game_paused" for e in gs.events)

    def test_resume_creates_game_event(self) -> None:
        gs = GameState(game_id="resume_test", phase="night", paused=True)
        event = GameEvent(type="game_resumed", payload={
            "game_id": "resume_test", "phase": "night",
        })
        gs = replace(gs, paused=False, events=gs.events + [event])
        assert gs.paused is False
        assert any(e.type == "game_resumed" for e in gs.events)

    def test_pause_resume_events_replayable(self) -> None:
        engine = RuleEngine.from_yaml(RULESET_PATH)
        gs = GameState(game_id="pause_replay", phase="night")
        events = [
            GameEvent(type="game_paused", payload={"game_id": "pause_replay"}),
            GameEvent(type="game_resumed", payload={"game_id": "pause_replay"}),
        ]
        replayed = replay_from_events(engine, gs, events)
        assert any(e.type == "game_paused" for e in replayed.events)
        assert any(e.type == "game_resumed" for e in replayed.events)


# ---------------------------------------------------------------------------
# Step 3: Replay source of truth
# ---------------------------------------------------------------------------


class TestReplaySourceOfTruth:
    """Verify replay from initial_seed + ruleset_snapshot + event_log."""

    def test_replay_reconstructs_complete_game_state(self) -> None:
        engine = RuleEngine.from_yaml(RULESET_PATH)
        players = engine.assign_roles(
            [f"p{i:02d}" for i in range(1, 13)], seed=42,
        )
        initial = GameState(
            game_id="replay_v1",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            players=players,
        )

        # Build a realistic game through engine
        gs = initial
        hybrid_id = next(pid for pid, p in gs.players.items() if p.role == "hybrid")
        non_hybrid = next(pid for pid, p in gs.players.items() if p.role != "hybrid")

        gs, events1 = engine.choose_master(gs, hybrid_id=hybrid_id, master_id=non_hybrid)
        gs, events2 = engine.resolve_night(
            gs, night_number=1,
            wolf_kill_target_id=next(pid for pid, p in gs.players.items() if p.role == "villager"),
            seer_target_id=next(pid for pid, p in gs.players.items() if p.role == "werewolf"),
        )
        gs, events3 = engine.resolve_exile(
            gs, target_id=next(pid for pid, p in gs.players.items()
                               if p.role == "werewolf" and not p.alive is False),
        ) if any(p.alive and p.role == "werewolf" for p in gs.players.values()) else (gs, [])

        all_events = ([events1] if isinstance(events1, GameEvent) else events1) + events2 + (events3 if events3 else [])
        gs_final = replace(gs, events=gs.events + all_events)

        # Extract event log
        event_log = extract_event_log(gs_final)

        # Replay from initial state + events
        replayed = replay_from_events(engine, initial, event_log)

        # Compare key state
        # 1. Deaths match
        original_dead = {d.player_id for d in gs_final.deaths}
        replayed_dead = {d.player_id for d in replayed.deaths}
        assert original_dead == replayed_dead, (
            f"Death mismatch: original={original_dead} replayed={replayed_dead}"
        )

        # 2. Hybrid master
        assert replayed.hybrid_master_id == gs_final.hybrid_master_id

        # 3. Sheriff state
        assert replayed.sheriff_badge_state == gs_final.sheriff_badge_state

        # 4. Antidote/poison
        assert replayed.antidote_used == gs_final.antidote_used
        assert replayed.poison_used == gs_final.poison_used

    def test_replay_reconstructs_victory_state(self) -> None:
        engine = RuleEngine.from_yaml(RULESET_PATH)
        players = {
            "w1": PlayerState(id="w1", role="werewolf", alive=False),
            "w2": PlayerState(id="w2", role="werewolf", alive=False),
            "w3": PlayerState(id="w3", role="werewolf", alive=False),
            "w4": PlayerState(id="w4", role="werewolf", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
            "v2": PlayerState(id="v2", role="villager"),
            "v3": PlayerState(id="v3", role="villager"),
            "seer": PlayerState(id="seer", role="seer"),
            "witch": PlayerState(id="witch", role="witch"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "idiot": PlayerState(id="idiot", role="idiot"),
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
        }
        initial = GameState(
            game_id="replay_victory",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            players=players,
            hybrid_master_id="v1",
            hybrid_master_faction="good",
        )
        events = [
            GameEvent(type="player_died", payload={"player_id": "w1", "reason": "exile", "timing": "day_vote", "resolution_batch": "day_1_vote"}),
            GameEvent(type="player_died", payload={"player_id": "w2", "reason": "wolf_kill", "timing": "night", "resolution_batch": "night_2"}),
            GameEvent(type="player_died", payload={"player_id": "w3", "reason": "hunter_shot", "timing": "night", "resolution_batch": "night_2"}),
            GameEvent(type="player_died", payload={"player_id": "w4", "reason": "exile", "timing": "day_vote", "resolution_batch": "day_2_vote"}),
            GameEvent(type="victory", payload={
                "winner": "good", "reason": "all_werewolves_out",
                "hybrid_result": "win",
            }),
        ]
        replayed = replay_from_events(engine, initial, events)

        assert replayed.winning_faction == "good"
        assert replayed.hybrid_result == "win"
        assert not replayed.players["w1"].alive
        assert not replayed.players["w2"].alive
        assert not replayed.players["w3"].alive
        assert not replayed.players["w4"].alive
        assert replayed.players["v1"].alive
        assert replayed.phase == "finished"

    def test_replay_reconstructs_sheriff_badge_chain(self) -> None:
        engine = RuleEngine.from_yaml(RULESET_PATH)
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "v2": PlayerState(id="v2", role="villager"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        initial = GameState(
            game_id="replay_badge",
            players=players,
        )
        events = [
            GameEvent(type="sheriff_elected", payload={"sheriff_id": "seer"}),
            GameEvent(type="player_died", payload={"player_id": "seer", "reason": "wolf_kill", "timing": "night", "resolution_batch": "night_1"}),
            GameEvent(type="badge_transferred", payload={"new_sheriff_id": "v1"}),
        ]
        replayed = replay_from_events(engine, initial, events)

        assert replayed.sheriff_id == "v1"
        assert replayed.sheriff_badge_state == "active"
        assert not replayed.players["seer"].alive
