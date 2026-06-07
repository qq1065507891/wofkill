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



RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _last_non_broadcast_event(gs: GameState) -> GameEvent:
    return next(e for e in reversed(gs.events) if e.type != "judge_broadcast")

def _make_scripted_state(
    *,
    wolf_kill_targets: list[str | None],
    exile_votes_list: list[dict[str, str]] | None = None,
) -> RuntimeState:
    return {
        "game_state": GameState(game_id="scripted01"),
        "engine": _new_engine(),
        "wolf_kill_targets": wolf_kill_targets,
        "wolf_kill_target_id": wolf_kill_targets[0] if wolf_kill_targets else None,
        "use_antidote": False,
        "poison_target_id": None,
        "seer_target_id": None,
        "hybrid_master_target_id": None,
        "self_destruct_wolf_id": None,
        "exile_votes": exile_votes_list[0] if exile_votes_list else {},
        "revote": False,
        "sheriff_candidates": [],
        "sheriff_votes": {},
        "sheriff_withdrawing": [],
        "badge_decision": "tear",
        "badge_target_id": None,
        "hunter_shot_target_id": None,
    }


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def test_graph_compiles() -> None:
    graph = build_game_graph()
    assert "setup_game" in graph.nodes
    assert "finish_game" in graph.nodes


def test_graph_compiles_with_checkpoint() -> None:
    cp = make_checkpointer()
    graph = build_game_graph_with_checkpoint(cp)
    assert "finish_game" in graph.nodes


# ---------------------------------------------------------------------------
# Setup + assign roles
# ---------------------------------------------------------------------------

def test_setup_and_assign_roles() -> None:
    graph = build_game_graph()
    engine = _new_engine()
    gs = GameState(game_id="test_setup")

    # Run just setup + assign by streaming and stopping early
    events_seen = []
    for chunk in graph.stream(
        {"game_state": gs, "engine": engine, "wolf_kill_target_id": None,
         "use_antidote": False, "poison_target_id": None},
        {"recursion_limit": 10},
    ):
        for node_name, output in chunk.items():
            events_seen.append(node_name)
            if node_name == "assign_roles":
                result_gs = output.get("game_state", gs)
                assert len(result_gs.players) == 12
                assert result_gs.phase == "roles_assigned"
                return

    pytest.fail("assign_roles node never reached")

def test_scripted_peace_game_does_not_forge_winner() -> None:
    """Peace game: no kills, no votes. Runtime must not invent a legal winner."""
    graph = build_game_graph()
    engine = _new_engine()
    state: RuntimeState = {
        "game_state": GameState(game_id="peace01"),
        "engine": engine,
        "wolf_kill_target_id": None,
        "use_antidote": False,
        "poison_target_id": None,
        "seer_target_id": None,
        "hybrid_master_target_id": None,
        "self_destruct_wolf_id": None,
        "exile_votes": {},
        "revote": False,
        "sheriff_candidates": [],
        "sheriff_votes": {},
        "sheriff_withdrawing": [],
        "badge_decision": "tear",
        "badge_target_id": None,
        "hunter_shot_target_id": None,
    }
    with pytest.raises(Exception) as exc_info:
        graph.invoke(state, {"recursion_limit": 80})
    assert "Recursion" in type(exc_info.value).__name__ or "recursion" in str(exc_info.value).lower()


def test_scripted_game_with_wolf_kill_does_not_forge_winner() -> None:
    """One scripted wolf kill is not enough to invent a terminal winner."""
    graph = build_game_graph()
    engine = _new_engine()
    # Pre-assign to know a non-wolf target
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
    non_wolf = next(pid for pid, p in players.items() if p.role == "villager")

    state: RuntimeState = {
        "game_state": GameState(game_id="kill01"),
        "engine": engine,
        "wolf_kill_target_id": non_wolf,
        "use_antidote": False,
        "poison_target_id": None,
        "seer_target_id": None,
        "hybrid_master_target_id": None,
        "self_destruct_wolf_id": None,
        "exile_votes": {},
        "revote": False,
        "sheriff_candidates": [],
        "sheriff_votes": {},
        "sheriff_withdrawing": [],
        "badge_decision": "tear",
        "badge_target_id": None,
        "hunter_shot_target_id": None,
    }
    with pytest.raises(Exception) as exc_info:
        graph.invoke(state, {"recursion_limit": 80})
    assert "Recursion" in type(exc_info.value).__name__ or "recursion" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Node-level unit tests (graph lifecycle)
# ---------------------------------------------------------------------------

def test_setup_game_node() -> None:
    from werewolf_agent.runtime.graph import setup_game
    result = setup_game({"game_state": None, "engine": None})
    gs = result["game_state"]
    assert gs.phase == "setup"
    assert result["engine"] is not None


def test_assign_roles_node() -> None:
    from werewolf_agent.runtime.graph import assign_roles
    engine = _new_engine()
    gs = GameState(game_id="test", phase="setup")
    result = assign_roles({"game_state": gs, "engine": engine})
    gs = result["game_state"]
    assert len(gs.players) == 12
    from collections import Counter
    roles = Counter(p.role for p in gs.players.values())
    assert roles["werewolf"] == 4
    assert roles["villager"] == 3


def test_enter_night_increments_night() -> None:
    from werewolf_agent.runtime.graph import enter_night
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="n1", players=players, phase="roles_assigned", night_number=0)
    result = enter_night({"game_state": gs, "engine": engine})
    assert result["game_state"].night_number == 1
    assert result["game_state"].phase == "night"

def test_reflection_node_persists_entries_to_repository() -> None:
    from werewolf_agent.runtime.nodes.summary import reflection

    class FakeRepository:
        def __init__(self) -> None:
            self.saved: list[dict[str, Any]] = []

        def load_all_reflections(self) -> list[dict[str, Any]]:
            return []

        def save_reflection(self, entry: dict[str, Any]) -> None:
            self.saved.append(dict(entry))

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=False),
    }
    gs = GameState(
        game_id="reflection_persist",
        players=players,
        winning_faction="good",
        day_number=3,
    )
    repo = FakeRepository()

    result = reflection({"game_state": gs, "engine": _new_engine(), "repository": repo})

    assert result["game_state"].events[-1].type == "reflection_complete"
    assert [entry["player_id"] for entry in repo.saved] == ["p01", "p02"]
    assert all(entry["game_id"] == "reflection_persist" for entry in repo.saved)

def test_single_wolf_vote_uses_global_agent_timeout(monkeypatch) -> None:
    engine = _new_engine()
    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = GameState(game_id="wolf_timeout_config", players=players, night_number=1)
    captured: dict[str, float] = {}

    class Agent:
        def act(self, context):
            return (
                PlayerAction(action_type=ActionType.WOLF_KILL, target_id="p02"),
                RetryInfo(),
            )

    class Registry:
        def get_agent(self, player_id):
            return Agent()

    def fake_timed_call(fn, *args, timeout, fallback=None):
        captured["timeout"] = timeout
        return fn(*args)

    monkeypatch.setattr("werewolf_agent.runtime.timers.timed_call", fake_timed_call)

    result = _single_wolf_vote(
        {
            "game_state": gs,
            "engine": engine,
            "agent_call_timeout": 120.0,
        },
        engine,
        Registry(),
        "p01",
    )

    assert captured["timeout"] == 180.0
    assert result["wolf_action"] == "kill"
    assert result["wolf_kill_target_id"] == "p02"

def test_dispatch_agent_direct_call_when_timeout_zero(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import _shared as shared_mod
    import time
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    def fake_agent_adapter(state, engine, registry, player_id):
        return {"ok": player_id}

    class Registry:
        def get_agent(self, player_id):
            return object()

    result = shared_mod._dispatch_agent(
        {
            "game_state": GameState(game_id="wait_before_request"),
            "engine": _new_engine(),
            "agent_registry": Registry(),
            "agent_call_timeout": 0,
        },
        fake_agent_adapter,
        "p01",
    )

    assert result == {"ok": "p01"}
    assert len(sleeps) == 0

def test_manual_timer_expiration_is_deterministic() -> None:
    from werewolf_agent.runtime.timers import ManualTimer

    timer = ManualTimer(expired_keys={"speech:p01"})

    assert timer.expired("speech:p01") is True
    assert timer.expired("speech:p02") is False


def test_action_trace_audit_flags_timeline_confusion() -> None:
    event = _action_trace_event(
        player_id="p01",
        phase="speech",
        action_trace={
            "raw_text": "我认为第一天警上之后，晚上才进入首夜验人。",
            "parsed_action": {"reason": "第一天之后才首夜"},
        },
    )

    assert event.payload["timeline_confusion"]
    assert event.payload["timeline_confusion"][0]["type"] == "first_night_after_first_day"


# ---------------------------------------------------------------------------
# Victory check
# ---------------------------------------------------------------------------

def test_check_victory_good_wins() -> None:
    from werewolf_agent.runtime.graph import check_victory
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    # Kill all wolves
    wolves = [pid for pid, p in players.items() if p.role == "werewolf"]
    dead_players = {pid: replace(players[pid], alive=False) for pid in wolves}
    all_players = {**players, **dead_players}
    gs = GameState(game_id="vw1", players=all_players)
    result = check_victory({"game_state": gs, "engine": engine})
    assert result["game_state"].winning_faction == "good"

def test_check_victory_no_winner_yet() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="vw0", players=players)
    result = check_victory({"game_state": gs, "engine": engine})
    assert result["game_state"].winning_faction is None

def test_check_victory_does_not_force_scripted_day_limit_winner() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="vw_day4", players=players, day_number=4)

    result = check_victory({"game_state": gs, "engine": engine})

    assert result["game_state"].winning_faction is None
    assert result["_victory_result"].winner is None

def test_stable_seed_is_deterministic_for_same_parts() -> None:
    assert _stable_seed("game-1", "roles") == _stable_seed("game-1", "roles")
    assert _stable_seed("game-1", "roles") != _stable_seed("game-1", "wolf", 1)


# ---------------------------------------------------------------------------
# Conditional edge tests (routing)
# ---------------------------------------------------------------------------

def test_route_victory_finishes_when_won() -> None:
    from werewolf_agent.runtime.graph import route_victory
    gs = GameState(winning_faction="good")
    assert route_victory({"game_state": gs}) == "finish_game"


def test_route_victory_continues_when_no_winner() -> None:
    from werewolf_agent.runtime.graph import route_victory
    gs = GameState()
    assert route_victory({"game_state": gs}) == "enter_night"


def test_route_after_vote_exile() -> None:
    from werewolf_agent.runtime.graph import route_after_vote
    gs = GameState(events=[GameEvent(type="vote_resolved", payload={"exiled": "p01", "reason": "majority"})])
    result = route_after_vote({"game_state": gs})
    assert result == "resolve_exile"


def test_route_after_vote_tie() -> None:
    from werewolf_agent.runtime.graph import route_after_vote
    gs = GameState(events=[GameEvent(type="vote_resolved", payload={"exiled": None, "reason": "first_tie_pk"})])
    result = route_after_vote({"game_state": gs})
    assert result == "tie_pk_speech"


def test_route_after_announce_day2_discussion() -> None:
    from werewolf_agent.runtime.graph import route_after_announce
    gs = GameState(day_number=2, night_number=2)
    assert route_after_announce({"game_state": gs}) == "free_discussion"


class TestRouteAfterAnnounceSheriffEntry:
    """fix-sheriff-announce-route: D1 night_death_last_words should route to sheriff_first_day_entry when no sheriff."""

    def _make_state(self, *, day_number: int, interrupt_count: int):
        from werewolf_agent.core.models import GameState, PlayerState
        from unittest.mock import MagicMock
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        gs = GameState(
            players=alive,
            day_number=day_number,
            night_number=1,
            sheriff_interrupt_count=interrupt_count,
        )
        engine = MagicMock()
        engine.check_victory.return_value = MagicMock(winner=None)
        return {"game_state": gs, "engine": engine}

    def test_d1_count_0_routes_to_sheriff(self):
        from werewolf_agent.runtime.graph import route_after_announce
        state = self._make_state(day_number=1, interrupt_count=0)
        result = route_after_announce(state)
        assert result == "sheriff_first_day_entry", (
            f"D1 count=0 must go to sheriff_first_day_entry, got {result}"
        )

    def test_d1_count_1_routes_to_sheriff(self):
        from werewolf_agent.runtime.graph import route_after_announce
        state = self._make_state(day_number=1, interrupt_count=1)
        result = route_after_announce(state)
        assert result == "sheriff_first_day_entry", (
            f"D1 count=1 (post self-destruct) must go to sheriff_first_day_entry, got {result}"
        )

    def test_d1_count_2_routes_to_discussion(self):
        from werewolf_agent.runtime.graph import route_after_announce
        state = self._make_state(day_number=1, interrupt_count=2)
        result = route_after_announce(state)
        assert result == "free_discussion", (
            f"D1 count=2 (badge torn) must go to free_discussion, got {result}"
        )

    def test_d2_count_0_routes_to_discussion(self):
        from werewolf_agent.runtime.graph import route_after_announce
        state = self._make_state(day_number=2, interrupt_count=0)
        result = route_after_announce(state)
        assert result == "free_discussion", (
            f"D2+ without sheriff should go to free_discussion, got {result}"
        )


def test_route_after_free_discussion_continues_until_speech_queue_done() -> None:
    from werewolf_agent.runtime.graph import route_self_destruct_check

    gs = GameState()

    assert route_self_destruct_check({
        "game_state": gs,
        "speech_order": ["p01", "p02"],
        "speech_index": 1,
        "current_speaker_id": "p02",
    }) == "continue_discussion"
    assert route_self_destruct_check({
        "game_state": gs,
        "speech_order": ["p01", "p02"],
        "speech_index": 2,
        "current_speaker_id": None,
    }) == "summarize_positions"


# ---------------------------------------------------------------------------
# fix-sheriff-entry: route_after_resolve_night must not unconditionally
# route to announce_deaths_with_badge_loss. The badge-loss variant emits
# "警徽因两度中断永久流失" which is only correct when the sheriff election
# was actually interrupted twice. For D1 N1 first resolve (count=0) we
# must use plain announce_deaths so the design doc flow
# (announce_deaths -> last_words -> sheriff_election) is preserved
# and the badge stays "none" — see commit 89b865b for the D1
# self-destruct fix that established this behavior.
# ---------------------------------------------------------------------------

def _make_resolve_night_state(*, day_number: int, interrupt_count: int) -> RuntimeState:
    """Build a minimal state for routing tests of route_after_resolve_night.

    12 alive players with the standard role mix so check_victory returns
    no winner and the routing decision depends only on sheriff fields.
    night_number mirrors day_number: on D1 we just finished N1, on D2 we
    just finished N2, etc. (D1 has interrupt_count=0, others may vary.)
    """
    roles = (
        ["werewolf"] * 4
        + ["villager"] * 3
        + ["seer", "witch", "hunter", "idiot", "hybrid"]
    )
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role=role, alive=True)
        for i, role in enumerate(roles, start=1)
    }
    gs = GameState(
        game_id=f"resolve_night_d{day_number}_i{interrupt_count}",
        players=players,
        day_number=day_number,
        night_number=day_number,
        sheriff_id=None,
        sheriff_badge_state="none",
        sheriff_interrupt_count=interrupt_count,
    )
    return {
        "game_state": gs,
        "engine": _new_engine(),
    }


def test_route_after_resolve_night_d1_interrupt0_routes_to_announce_deaths() -> None:
    """D1 N1 first resolve, no sheriff, interrupt_count=0 → announce_deaths.

    Regression (fix-sheriff-entry): previously routed to
    announce_deaths_with_badge_loss, which unconditionally emitted
    '警徽因两度中断永久流失' even though the count was 0 and the badge
    should remain "none" until a real election interruption occurs.
    """
    state = _make_resolve_night_state(day_number=1, interrupt_count=0)
    result = route_after_resolve_night(state)
    assert result == "announce_deaths", (
        f"D1 N1 first resolve with interrupt_count=0 must go to "
        f"announce_deaths (preserving the design doc flow), got {result!r}"
    )


def test_route_after_resolve_night_d1_interrupt2_routes_to_badge_loss() -> None:
    """D1 N1 with interrupt_count=2 (two previous election interruptions) →
    announce_deaths_with_badge_loss. Preserves the actual 2-interrupt case.
    """
    state = _make_resolve_night_state(day_number=1, interrupt_count=2)
    result = route_after_resolve_night(state)
    assert result == "announce_deaths_with_badge_loss", (
        f"D1 with interrupt_count=2 should go to badge_loss, got {result!r}"
    )


def test_route_after_resolve_night_d2_no_sheriff_routes_to_announce_deaths() -> None:
    """D2+ resolve, no sheriff, interrupt_count=0 → announce_deaths.
    Sheriff election only happens on D1.
    """
    state = _make_resolve_night_state(day_number=2, interrupt_count=0)
    result = route_after_resolve_night(state)
    assert result == "announce_deaths", (
        f"D2 without sheriff should announce deaths, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Replay tests
# ---------------------------------------------------------------------------

def test_replay_from_events_matches_state() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
    initial = GameState(game_id="replay01", players=players)
    events = [
        GameEvent(type="player_died", payload={"player_id": "p01", "reason": "wolf_kill", "timing": "night"}),
        GameEvent(type="sheriff_elected", payload={"sheriff_id": "p03"}),
        GameEvent(type="badge_torn", payload={}),
    ]
    replayed = replay_from_events(engine, initial, events)
    assert replayed.players["p01"].alive is False
    assert replayed.sheriff_badge_state == "torn"


def test_extract_event_log() -> None:
    gs = GameState(events=[GameEvent(type="test", payload={})])
    log = extract_event_log(gs)
    assert len(log) == 1


# ---------------------------------------------------------------------------
# Phase 1 regression
# ---------------------------------------------------------------------------

def test_phase1_rule_tests_still_pass() -> None:
    """Meta-test: confirm we didn't break Phase 1.

    Note: this test runs the rules tests in a subprocess.  The
    project pytest.ini sets ``addopts = -n --dist=loadfile`` (xdist
    parallelism) but xdist is not always installed in dev envs.
    The subprocess inherits pytest.ini, so without
    ``--override-ini="addopts="`` the subprocess fails with
    "unrecognized arguments: -n".  Force-disable addopts for the
    subprocess so the test works in both xdist-enabled and xdist-free
    environments.
    """
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/rules/test_rule_engine_v1.py", "-q",
         "--override-ini=addopts="],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"Phase 1 tests failed: {result.stdout[:500]} "
        f"(stderr: {result.stderr[:200]})"
    )