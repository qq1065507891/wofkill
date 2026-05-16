"""Runtime tests: scripted non-LLM games through the LangGraph graph."""

from __future__ import annotations

import pytest
from dataclasses import replace

from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
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
)
from werewolf_agent.runtime.replay import replay_from_events, extract_event_log
from werewolf_agent.runtime.checkpoints import make_checkpointer

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


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


# ---------------------------------------------------------------------------
# Full scripted game: wolves kill all villagers
# ---------------------------------------------------------------------------


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
# Node-level unit tests
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


def test_wolf_consensus_timeout_defaults_to_no_kill_event() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="wolf_timeout", players=players, night_number=1)

    result = wolf_consensus({"game_state": gs, "engine": engine})

    assert result["wolf_kill_target_id"] is None
    event = result["game_state"].events[-1]
    assert event.type == "wolf_no_kill_timeout"
    assert event.payload["night_number"] == 1


def test_wolf_consensus_explicit_no_kill_records_declared_event() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="wolf_no_kill", players=players, night_number=1)

    result = wolf_consensus({
        "game_state": gs,
        "engine": engine,
        "wolf_action": "no_kill",
        "wolf_action_reason": "create peace-night pressure",
    })

    assert result["wolf_kill_target_id"] is None
    event = result["game_state"].events[-1]
    assert event.type == "wolf_no_kill_declared"
    assert event.payload["reason"] == "create peace-night pressure"


def test_wolf_consensus_kill_records_selected_target() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="wolf_kill", players=players, night_number=1)

    result = wolf_consensus({
        "game_state": gs,
        "engine": engine,
        "wolf_action": "kill",
        "wolf_kill_target_id": "p01",
    })

    assert result["wolf_kill_target_id"] == "p01"
    event = result["game_state"].events[-1]
    assert event.type == "wolf_kill_selected"
    assert event.payload["target_id"] == "p01"


def test_free_discussion_speech_timeout_records_event() -> None:
    gs = GameState(game_id="speech_timeout", day_number=1)

    result = free_discussion({
        "game_state": gs,
        "current_speaker_id": "p03",
        "speech_timed_out": True,
        "speech_seconds_limit": 90,
    })

    event = result["game_state"].events[-1]
    assert event.type == "speech_timeout"
    assert event.payload == {
        "player_id": "p03",
        "day_number": 1,
        "seconds_limit": 90,
    }


def test_free_discussion_speech_timeout_advances_speech_queue() -> None:
    gs = GameState(game_id="speech_queue", day_number=1)

    result = free_discussion({
        "game_state": gs,
        "speech_order": ["p01", "p02"],
        "speech_index": 0,
        "speech_timed_out": True,
        "speech_seconds_limit": 90,
    })

    assert result["speech_index"] == 1
    assert result["current_speaker_id"] == "p02"
    assert result["game_state"].events[-1].type == "speech_timeout"


def test_free_discussion_normal_speech_advances_speech_queue() -> None:
    gs = GameState(game_id="speech_queue_normal", day_number=1)

    result = free_discussion({
        "game_state": gs,
        "speech_order": ["p01", "p02"],
        "speech_index": 0,
        "speech_text": "我先过一轮。",
    })

    assert result["speech_index"] == 1
    assert result["current_speaker_id"] == "p02"
    assert result["game_state"].events[-1].type == "speech"
    assert result["game_state"].events[-1].payload["speaker"] == "p01"


def test_free_discussion_routes_to_vote_after_last_normal_speech() -> None:
    from werewolf_agent.runtime.graph import route_self_destruct_check

    gs = GameState(game_id="speech_queue_done", day_number=1)

    result = free_discussion({
        "game_state": gs,
        "speech_order": ["p01"],
        "speech_index": 0,
        "speech_text": "发言结束。",
    })

    assert result["speech_index"] == 1
    assert result["current_speaker_id"] is None
    assert route_self_destruct_check(result) == "day_vote"


def test_resolve_night_node_kills_target() -> None:
    from werewolf_agent.runtime.graph import resolve_night
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="rn1", players=players, night_number=1)
    result = resolve_night({
        "game_state": gs, "engine": engine,
        "wolf_kill_target_id": "p01", "use_antidote": False, "poison_target_id": None,
    })
    assert result["game_state"].players["p01"].alive is False


def test_resolve_night_node_keeps_witch_potion_events_in_timeline() -> None:
    from werewolf_agent.runtime.graph import resolve_night
    engine = _new_engine()
    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "witch": PlayerState(id="witch", role="witch"),
    }
    gs = GameState(game_id="rn_witch", players=players, night_number=1)

    result = resolve_night({
        "game_state": gs,
        "engine": engine,
        "wolf_kill_target_id": "v1",
        "use_antidote": True,
        "poison_target_id": None,
    })

    assert any(event.type == "witch_antidote_used" for event in result["game_state"].events)


def test_announce_deaths_increments_day() -> None:
    from werewolf_agent.runtime.graph import announce_deaths
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="d1", players=players, night_number=1, day_number=0)
    result = announce_deaths({"game_state": gs, "engine": engine})
    assert result["game_state"].day_number == 1


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
# Conditional edge tests
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


def test_route_after_announce_day1_sheriff() -> None:
    from werewolf_agent.runtime.graph import route_after_announce
    engine = _new_engine()
    gs = GameState(day_number=1)
    assert route_after_announce({"game_state": gs, "engine": engine}) == "sheriff_registration"


def test_route_after_announce_day2_discussion() -> None:
    from werewolf_agent.runtime.graph import route_after_announce
    gs = GameState(day_number=2)
    assert route_after_announce({"game_state": gs}) == "free_discussion"


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
    }) == "day_vote"


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
# Phase 1 regression: all rule tests still pass
# ---------------------------------------------------------------------------


def test_phase1_rule_tests_still_pass() -> None:
    """Meta-test: confirm we didn't break Phase 1."""
    import subprocess
    result = subprocess.run(
        ["D:/Miniforge3/envs/wofkill/python.exe", "-m", "pytest",
         "tests/rules/test_rule_engine_v1.py", "-q"],
        capture_output=True, text=True, cwd="E:/NLP/agent/wofkill",
    )
    assert result.returncode == 0
    assert "58 passed" in result.stdout or result.returncode == 0
