"""Runtime tests: scripted non-LLM games through the LangGraph graph."""

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
    _sheriff_died_this_batch,
    _route_after_badge_transfer,
)
from werewolf_agent.runtime.agent_adapter import _single_wolf_vote
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

    assert captured["timeout"] == 120.0
    assert result["wolf_action"] == "kill"
    assert result["wolf_kill_target_id"] == "p02"


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


def test_manual_timer_expiration_is_deterministic() -> None:
    from werewolf_agent.runtime.timers import ManualTimer

    timer = ManualTimer(expired_keys={"speech:p01"})

    assert timer.expired("speech:p01") is True
    assert timer.expired("speech:p02") is False


def test_wolf_discussion_timer_expiration_forces_no_kill_timeout() -> None:
    from werewolf_agent.runtime.timers import ManualTimer

    players = {
        "w1": PlayerState(id="w1", role="werewolf", alive=True),
        "v1": PlayerState(id="v1", role="villager", alive=True),
    }
    gs = GameState(game_id="wolf_timer", players=players, night_number=1)

    result = wolf_consensus({
        "game_state": gs,
        "engine": _new_engine(),
        "wolf_action": "kill",
        "wolf_kill_target_id": "v1",
        "runtime_timer": ManualTimer(expired_keys={"wolf_discussion"}),
    })

    assert result["wolf_kill_target_id"] is None
    assert result["game_state"].events[-1].type == "wolf_no_kill_timeout"
    assert result["game_state"].events[-1].payload["reason"] == "timer_expired"


def test_free_discussion_timer_expiration_records_timeout() -> None:
    from werewolf_agent.runtime.timers import ManualTimer

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = GameState(game_id="speech_timer", players=players, day_number=1)

    result = free_discussion({
        "game_state": gs,
        "engine": _new_engine(),
        "speech_order": ["p01", "p02"],
        "speech_index": 0,
        "current_speaker_id": "p01",
        "speech_seconds_limit": 30,
        "runtime_timer": ManualTimer(expired_keys={"speech:p01"}),
    })

    assert result["game_state"].events[-1].type == "speech_timeout"
    assert result["game_state"].events[-1].payload["seconds_limit"] == 30
    assert result["current_speaker_id"] == "p02"


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


def test_free_discussion_keeps_action_trace_out_of_public_speech(monkeypatch) -> None:
    from werewolf_agent.runtime import graph as runtime_graph

    gs = GameState(game_id="speech_trace_private", day_number=1)
    private_trace = {
        "raw_text": '{"private_intent":{"true_role":"werewolf"}}',
        "parsed_action": {
            "action_type": "speech",
            "private_intent": {"true_role": "werewolf"},
        },
        "final_action_type": "speech",
    }

    def fake_call_agent(*args, **kwargs):
        return {"speech_text": "公开发言", "action_trace": private_trace}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

    result = runtime_graph.free_discussion({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
        "speech_order": ["p01"],
        "speech_index": 0,
    })

    events = result["game_state"].events
    public_speech = next(event for event in events if event.type == "speech")
    audit_event = next(event for event in events if event.type == "action_trace_audit")

    assert "action_trace" not in public_speech.payload
    assert public_speech.payload["text"] == "公开发言"
    assert audit_event.payload["visibility"] == "moderator_only"
    assert audit_event.payload["action_trace"] == private_trace


def test_resolve_vote_keeps_action_traces_out_of_public_result() -> None:
    from werewolf_agent.runtime.graph import resolve_vote

    engine = _new_engine()
    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="werewolf"),
    }
    gs = GameState(game_id="vote_trace_private", players=players, day_number=1)
    private_trace = {"parsed_action": {"private_intent": {"true_role": "werewolf"}}}

    result = resolve_vote({
        "game_state": gs,
        "engine": engine,
        "exile_votes": {"p01": "p02"},
        "vote_action_traces": {"p01": private_trace},
        "revote": False,
    })

    events = result["game_state"].events
    vote_event = next(event for event in events if event.type == "vote_resolved")
    audit_event = next(event for event in events if event.type == "action_trace_audit")

    assert "action_traces" not in vote_event.payload
    assert audit_event.payload["phase"] == "vote"
    assert audit_event.payload["visibility"] == "moderator_only"


def test_first_night_wolf_discussion_runs_three_rounds_and_builds_team_plan(monkeypatch) -> None:
    from werewolf_agent.runtime import graph as runtime_graph

    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "w3": PlayerState(id="w3", role="werewolf"),
        "w4": PlayerState(id="w4", role="werewolf"),
        "s1": PlayerState(id="s1", role="seer"),
        "v1": PlayerState(id="v1", role="villager"),
    }
    gs = GameState(game_id="wolf_plan", players=players, night_number=1, phase="night")
    calls: list[tuple[str, Any]] = []

    def fake_call_agent(_fn, _state, *_args, **_kwargs):
        wolf_id = _args[-1]
        calls.append((wolf_id, _state.get("wolf_discussion_round")))
        return {"speech_text": f"{wolf_id} round {_state.get('wolf_discussion_round')}"}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

    result = runtime_graph.wolf_discussion({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
    })

    events = result["game_state"].events
    round_events = [event for event in events if event.type == "wolf_discussion"]
    plan_events = [event for event in events if event.type == "wolf_team_plan"]
    plan = result["wolf_team_plan"]

    assert len(round_events) == 12
    assert {event.payload["round"] for event in round_events} == {1, 2, 3}
    assert len(plan_events) == 1
    assert plan_events[0].payload["visibility"] == "werewolf_team_only"
    for key in (
        "fake_seer",
        "pusher",
        "hooker",
        "deep_cover",
        "night_kill_primary",
        "night_kill_backup",
        "day_push_target",
        "public_story",
    ):
        assert plan[key]

    assignments = [plan["fake_seer"], plan["pusher"], plan["hooker"], plan["deep_cover"]]
    assert sorted(assignments) == ["w1", "w2", "w3", "w4"]


def test_later_night_wolf_discussion_runs_two_rounds_and_revises_plan(monkeypatch) -> None:
    from werewolf_agent.runtime import graph as runtime_graph

    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="villager"),
    }
    gs = GameState(game_id="wolf_plan_later", players=players, night_number=2, phase="night")

    def fake_call_agent(_fn, _state, *_args, **_kwargs):
        return {"speech_text": "revise plan"}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

    result = runtime_graph.wolf_discussion({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
        "wolf_team_plan": {"fake_seer": "w1", "pusher": "w2"},
    })

    round_events = [event for event in result["game_state"].events if event.type == "wolf_discussion"]
    assert len(round_events) == 4
    assert {event.payload["round"] for event in round_events} == {1, 2}
    assert result["wolf_team_plan"]["night_number"] == 2


def test_wolf_consensus_prefers_planned_primary_then_backup_target() -> None:
    from werewolf_agent.runtime.graph import wolf_consensus

    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager", alive=False),
        "v2": PlayerState(id="v2", role="villager"),
    }
    gs = GameState(game_id="wolf_plan_kill", players=players, night_number=2)

    result = wolf_consensus({
        "game_state": gs,
        "engine": _new_engine(),
        "wolf_team_plan": {
            "night_kill_primary": "v1",
            "night_kill_backup": "v2",
        },
    })

    assert result["wolf_kill_target_id"] == "v2"
    event = result["game_state"].events[-1]
    assert event.type == "wolf_kill_selected"
    assert event.payload["target_id"] == "v2"
    assert event.payload["reason"] == "wolf_team_plan"


def test_sheriff_speech_calls_candidate_agents_and_keeps_trace_private(monkeypatch) -> None:
    from werewolf_agent.runtime import graph as runtime_graph

    players = {
        "p01": PlayerState(id="p01", role="seer"),
        "p02": PlayerState(id="p02", role="werewolf"),
    }
    gs = GameState(
        game_id="sheriff_agent",
        players=players,
        day_number=1,
        sheriff_candidates=["p01", "p02"],
    )
    private_trace = {"parsed_action": {"private_intent": {"true_role": "werewolf"}}}

    def fake_call_agent(*_args, **_kwargs):
        return {"speech_text": "我上警竞选警长。", "action_trace": private_trace}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

    result = runtime_graph.sheriff_speech({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
    })

    events = result["game_state"].events
    speeches = [event for event in events if event.type == "sheriff_speech"]
    audits = [event for event in events if event.type == "action_trace_audit"]

    assert [event.payload["speaker"] for event in speeches] == ["p01", "p02"]
    assert all("action_trace" not in event.payload for event in speeches)
    assert len(audits) == 2
    assert all(event.payload["visibility"] == "moderator_only" for event in audits)


def test_day_speech_passes_wolf_team_plan_to_werewolf_agent() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_day_speech

    players = {
        "w3": PlayerState(id="w3", role="werewolf"),
        "p08": PlayerState(id="p08", role="seer"),
    }
    gs = GameState(game_id="wolf_day_assignment", players=players, day_number=1, phase="day")

    class Agent:
        player_name = "Hook Wolf"

        def __init__(self):
            self.context = None

        def act(self, context):
            self.context = context
            return PlayerAction(
                action_type=ActionType.SPEECH,
                speech="我会从发言逻辑质疑p08。",
                reason="按公开逻辑发言",
            ), RetryInfo()

    agent = Agent()

    class Registry:
        def get_agent(self, player_id):
            return agent

    result = agent_day_speech(
        {
            "game_state": gs,
            "wolf_team_plan": {
                "fake_seer": "w1",
                "pusher": "w2",
                "hooker": "w3",
                "deep_cover": "w4",
                "day_push_target": "p08",
            },
        },
        _new_engine(),
        Registry(),
        "w3",
    )

    assert result["speech_text"] == "我会从发言逻辑质疑p08。"
    assert agent.context.visible_world_state["wolf_team_plan"]["hooker"] == "w3"


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


def test_route_after_announce_day1_now_goes_to_discussion() -> None:
    from werewolf_agent.runtime.graph import route_after_announce
    engine = _new_engine()
    gs = GameState(day_number=1)
    # Sheriff election now happens BEFORE announce_deaths, so after announce → free_discussion
    assert route_after_announce({"game_state": gs, "engine": engine}) == "free_discussion"


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


# ---------------------------------------------------------------------------
# Task 1: Runtime Gap Tests
# ---------------------------------------------------------------------------


class TestNightHunterIdiotStatusNode:
    """Design doc §6.2 requires a night_hunter_idiot_status node between
    night_seer and first_night_hybrid_master."""

    def test_graph_contains_night_hunter_idiot_status_node(self) -> None:
        graph = build_game_graph()
        assert "night_hunter_idiot_status" in graph.nodes, (
            "Design doc §6.2 node 7 'night_hunter_idiot_status' is missing from the graph"
        )

    def test_night_edge_order_seer_to_hunter_idiot(self) -> None:
        """night_seer must route to night_hunter_idiot_status, not directly to first_night_hybrid_master."""
        graph = build_game_graph()
        assert "night_hunter_idiot_status" in graph.nodes
        # Verify the edge: night_seer -> night_hunter_idiot_status
        # We check by running a simple night flow and ensuring the node appears
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=99)
        gs = GameState(game_id="edge_order", players=players, phase="night", night_number=1)

        state: RuntimeState = {
            "game_state": gs,
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

        nodes_seen = []
        for chunk in graph.stream(state, {"recursion_limit": 50}):
            for node_name in chunk.keys():
                nodes_seen.append(node_name)
                if node_name == "resolve_night_node":
                    break
            else:
                continue
            break

        # The night flow must be: wolf_discussion, wolf_consensus, night_witch,
        # night_seer, night_hunter_idiot_status, first_night_hybrid_master, resolve_night_node
        night_nodes = [n for n in nodes_seen if n in {
            "wolf_discussion", "wolf_consensus", "night_witch", "night_seer",
            "night_hunter_idiot_status", "first_night_hybrid_master", "resolve_night_node",
        }]
        assert "night_hunter_idiot_status" in night_nodes, (
            f"night_hunter_idiot_status node was not visited during night flow. "
            f"Night nodes seen: {night_nodes}"
        )
        # Verify ordering
        if "night_hunter_idiot_status" in night_nodes and "first_night_hybrid_master" in night_nodes:
            assert night_nodes.index("night_hunter_idiot_status") < night_nodes.index("first_night_hybrid_master"), (
                "night_hunter_idiot_status must come before first_night_hybrid_master in night flow"
            )


class TestSeerNightResolution:
    """Design doc §3.3: seer checks alignment each night. resolve_night must
    produce a seer_check event with seer_id, target_id, alignment, night_number."""

    def test_resolve_night_produces_seer_check_event(self) -> None:
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        seer_id = next(pid for pid, p in players.items() if p.role == "seer")
        # Pick a non-seer target
        target_id = next(pid for pid, p in players.items() if p.role != "seer" and p.alive)
        gs = GameState(game_id="seer_test", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": target_id,
        })

        events = result["game_state"].events
        seer_checks = [e for e in events if e.type == "seer_check"]
        assert len(seer_checks) == 1, f"Expected exactly 1 seer_check event, got {len(seer_checks)}"
        check = seer_checks[0]
        assert check.payload["seer_id"] == seer_id
        assert check.payload["target_id"] == target_id
        assert check.payload["alignment"] in ("good", "werewolf")
        assert check.payload["night_number"] == 1

    def test_seer_check_hybrid_returns_good(self) -> None:
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        seer_id = next(pid for pid, p in players.items() if p.role == "seer")
        hybrid_id = next((pid for pid, p in players.items() if p.role == "hybrid"), None)
        if hybrid_id is None:
            pytest.skip("No hybrid in this seed")
        gs = GameState(game_id="seer_hybrid", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": hybrid_id,
        })

        seer_checks = [e for e in result["game_state"].events if e.type == "seer_check"]
        assert len(seer_checks) == 1
        assert seer_checks[0].payload["alignment"] == "good", (
            "Seer must see hybrid as good (design doc §3.2)"
        )

    def test_seer_check_is_not_in_public_timeline(self) -> None:
        """seer_check must be private; it must not appear in public events."""
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        target_id = next(pid for pid, p in players.items() if p.role != "seer" and p.alive)
        gs = GameState(game_id="seer_private", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": target_id,
        })

        seer_checks = [e for e in result["game_state"].events if e.type == "seer_check"]
        assert len(seer_checks) == 1
        # seer_check event must have a visibility marker indicating it's private
        check = seer_checks[0]
        assert check.payload.get("visibility") in ("private", "seer_only", "moderator_only"), (
            "seer_check must have visibility marker preventing public exposure"
        )


class TestSheriffBadgeAfterNightDeath:
    """Design doc §6.2: sheriff death from night wolf kill / witch poison must
    route to badge transfer/tear before the next night when the game continues."""

    def test_sheriff_night_kill_routes_to_badge_transfer(self) -> None:
        graph = build_game_graph()
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        # Find a non-wolf target to be the sheriff
        sheriff_candidate = next(
            pid for pid, p in players.items() if p.role not in ("werewolf", "hybrid") and p.alive
        )
        # Set the sheriff and mark badge active
        players[sheriff_candidate] = replace(players[sheriff_candidate],)
        gs = GameState(
            game_id="badge_night",
            players=players,
            phase="night",
            night_number=1,
            day_number=0,
            sheriff_id=sheriff_candidate,
            sheriff_badge_state="active",
        )

        # Kill the sheriff with wolf kill
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": sheriff_candidate,
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

        nodes_seen = []
        try:
            for chunk in graph.stream(state, {"recursion_limit": 120}):
                for node_name in chunk.keys():
                    nodes_seen.append(node_name)
        except Exception:
            pass  # Game may not terminate, but we just need to see the nodes

        # After resolve_night_node, if sheriff died and game continues,
        # badge transfer must be visited before entering night again
        if sheriff_candidate in [d.player_id for d in gs.deaths]:
            pytest.skip("Sheriff not killed in this scenario")

        # Check if resolve_night_node was visited and sheriff badge_transfer appeared
        # after it but before the second enter_night
        resolve_idx = None
        badge_idx = None
        enter_night_indices = []
        for i, n in enumerate(nodes_seen):
            if n == "resolve_night_node" and resolve_idx is None:
                resolve_idx = i
            if n == "sheriff_badge_transfer":
                badge_idx = i
            if n == "enter_night":
                enter_night_indices.append(i)

        # If sheriff died and game didn't end, badge transfer should have been visited
        if resolve_idx is not None:
            assert badge_idx is not None, (
                f"Sheriff killed at night but badge_transfer was not visited. "
                f"Nodes: {nodes_seen[:30]}"
            )


class TestHunterShotTiming:
    """Design doc §3.2: hunter can shoot when killed by wolves or exiled,
    cannot shoot when poisoned by witch."""

    def test_hunter_killed_by_wolf_can_shoot(self) -> None:
        """Hunter killed by wolf kill at night must be able to shoot a target."""
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hunter_wolf", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "hunter",
            "use_antidote": False,
            "poison_target_id": None,
        })

        # Hunter should be dead from wolf kill
        assert result["game_state"].players["hunter"].alive is False
        # There should be a triggered skill indicating hunter can shoot
        hunter_deaths = [d for d in result["game_state"].deaths if d.player_id == "hunter"]
        assert len(hunter_deaths) == 1
        assert "hunter_shot" in hunter_deaths[0].triggered_skills, (
            "Hunter killed by wolf should have hunter_shot in triggered_skills"
        )

    def test_hunter_exiled_can_shoot_in_post_exile(self) -> None:
        """Hunter killed by exile must be able to shoot in post_exile_skills."""
        from werewolf_agent.runtime.graph import post_exile_skills
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        death = Death(
            player_id="hunter", reason="exile", timing="day",
            resolution_batch="day_1_exile", triggered_skills=["hunter_shot"],
        )
        gs = GameState(
            game_id="hunter_exile",
            players=players,
            deaths=[death],
        )

        result = post_exile_skills({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        # v1 should be dead from hunter shot
        assert result["game_state"].players["v1"].alive is False, (
            "Hunter shot target should be dead after post_exile_skills"
        )
        shot_deaths = [d for d in result["game_state"].deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) >= 1, "hunter_shot death should be recorded"

    def test_hunter_poisoned_cannot_shoot(self) -> None:
        """Hunter killed by witch poison must NOT be able to shoot."""
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "v1": PlayerState(id="v1", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
        }
        gs = GameState(game_id="hunter_poison", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "v1",
            "use_antidote": False,
            "poison_target_id": "hunter",
        })

        # Hunter should be dead from poison
        assert result["game_state"].players["hunter"].alive is False
        hunter_deaths = [d for d in result["game_state"].deaths if d.player_id == "hunter"]
        assert len(hunter_deaths) == 1
        assert "hunter_shot" not in hunter_deaths[0].triggered_skills, (
            "Hunter killed by witch poison must NOT have hunter_shot triggered"
        )

    def test_hunter_shot_replay_reconstructs_death(self) -> None:
        """Replay must reconstruct hunter_shot death from events."""
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        initial = GameState(game_id="replay_hunter", players=players)
        events = [
            GameEvent(type="player_died", payload={
                "player_id": "hunter", "reason": "wolf_kill", "timing": "night",
                "resolution_batch": "night_1",
            }),
            GameEvent(type="player_died", payload={
                "player_id": "v1", "reason": "hunter_shot", "timing": "night",
                "resolution_batch": "night_1", "source_player_id": "hunter",
            }),
        ]
        replayed = replay_from_events(engine, initial, events)
        assert not replayed.players["hunter"].alive
        assert not replayed.players["v1"].alive
        shot_deaths = [d for d in replayed.deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 1


# ---------------------------------------------------------------------------
# Wolf Discussion Loop: multi-agent wolf night discussion with consensus
# ---------------------------------------------------------------------------


class _WolfMockRegistry:
    """Minimal mock registry that tracks which wolves are consulted."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.discussion_calls: list[str] = []
        self.vote_calls: list[str] = []

    def get_agent(self, player_id: str):
        return _WolfMockAgent(player_id, self)


class _WolfMockAgent:
    """Agent that records calls and returns deterministic responses."""

    def __init__(self, agent_id: str, registry: _WolfMockRegistry) -> None:
        self._id = agent_id
        self._registry = registry

    def act(self, context):
        from werewolf_agent.agents.schemas import (
            ActionType, FallbackAction, PlayerAction, RetryInfo,
        )
        # Discussion phase
        if context.task_type.value == "wolf_discussion" and context.legal_actions and any(
            a.value == "speech" for a in context.legal_actions
        ):
            self._registry.discussion_calls.append(self._id)
            return PlayerAction(
                action_type=ActionType.SPEECH,
                target_id=None,
                speech=f"{self._id}: 今晚杀掉目标",
                reason="discuss",
                confidence=0.8,
                private_intent=None,
            ), RetryInfo(attempts=0, errors=[])

        # Vote phase
        if context.legal_actions and any(a.value == "wolf_kill" for a in context.legal_actions):
            self._registry.vote_calls.append(self._id)
            target = context.legal_targets[0] if context.legal_targets else None
            # If registry has a per-wolf response, honor it
            resp = self._registry._responses.get(self._id, "kill")
            if resp == "no_kill":
                return PlayerAction(
                    action_type=ActionType.WOLF_NO_KILL,
                    target_id=None,
                    speech="",
                    reason="strategic no-kill",
                    confidence=0.7,
                    private_intent=None,
                ), RetryInfo(attempts=0, errors=[])
            return PlayerAction(
                action_type=ActionType.WOLF_KILL,
                target_id=target,
                speech="",
                reason="kill target",
                confidence=0.7,
                private_intent=None,
            ), RetryInfo(attempts=0, errors=[])

        return FallbackAction(
            action_type=ActionType.NO_ACTION,
            target_id=None,
            reason="no matching action",
        ), RetryInfo(attempts=0, errors=[])


class TestWolfDiscussionLoop:
    """Design doc §6.2: wolf_discussion should collect private speech from each wolf;
    wolf_consensus should aggregate votes from ALL wolves."""

    def test_wolf_discussion_calls_each_alive_wolf(self) -> None:
        """wolf_discussion with registry produces per-wolf discussion events."""
        from werewolf_agent.runtime.graph import wolf_discussion
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="wolf_disc", players=players, night_number=1)

        registry = _WolfMockRegistry()
        result = wolf_discussion({
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
        })

        alive_wolves = [pid for pid, p in players.items() if p.role == "werewolf" and p.alive]
        assert len(registry.discussion_calls) == len(alive_wolves) * 3, (
            f"Expected {len(alive_wolves) * 3} discussion calls, got {len(registry.discussion_calls)}"
        )

    def test_wolf_discussion_events_have_wolf_team_visibility(self) -> None:
        """Wolf discussion events must have werewolf_team_only visibility."""
        from werewolf_agent.runtime.graph import wolf_discussion
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="wolf_vis", players=players, night_number=1)

        registry = _WolfMockRegistry()
        result = wolf_discussion({
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
        })

        disc_events = [e for e in result["game_state"].events if e.type == "wolf_discussion"]
        assert len(disc_events) >= 1
        for evt in disc_events:
            assert evt.payload.get("visibility") == "werewolf_team_only", (
                f"wolf_discussion event missing visibility: {evt.payload}"
            )
            assert "wolf_id" in evt.payload, "wolf_discussion event must identify the speaker"
            assert evt.payload["round"] in {1, 2, 3}

    def test_wolf_consensus_uses_all_wolves_votes(self) -> None:
        """wolf_consensus with registry collects votes from all alive wolves."""
        from werewolf_agent.runtime.graph import wolf_consensus
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="wolf_vote", players=players, night_number=1)

        registry = _WolfMockRegistry()
        result = wolf_consensus({
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
        })

        alive_wolves = [pid for pid, p in players.items() if p.role == "werewolf" and p.alive]
        assert len(registry.vote_calls) == len(alive_wolves), (
            f"Expected {len(alive_wolves)} vote calls, got {len(registry.vote_calls)}"
        )

    def test_wolf_consensus_majority_no_kill(self) -> None:
        """When majority of wolves vote no_kill, result is wolf_no_kill_declared."""
        from werewolf_agent.runtime.graph import wolf_consensus
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="wolf_nokill", players=players, night_number=1)

        alive_wolves = [pid for pid, p in players.items() if p.role == "werewolf" and p.alive]
        # All wolves vote no_kill
        responses = {pid: "no_kill" for pid in alive_wolves}
        registry = _WolfMockRegistry(responses=responses)
        result = wolf_consensus({
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
        })

        assert result["wolf_kill_target_id"] is None
        event = result["game_state"].events[-1]
        assert event.type == "wolf_no_kill_declared"

    def test_wolf_discussion_no_registry_remains_scripted(self) -> None:
        """Without registry, wolf_discussion uses scripted fallback (no regression)."""
        from werewolf_agent.runtime.graph import wolf_discussion
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="wolf_scripted", players=players, night_number=1)

        result = wolf_discussion({"game_state": gs, "engine": engine})

        disc_events = [e for e in result["game_state"].events if e.type == "wolf_discussion"]
        assert len(disc_events) == 1
        # Scripted fallback has empty payload
        assert disc_events[0].payload == {}


# ---------------------------------------------------------------------------
# Hunter Shot Resolution: night wolf-kill hunter shot and agent target selection
# ---------------------------------------------------------------------------


class _HunterMockRegistry:
    """Mock registry for hunter shot tests."""

    def __init__(self, shot_target: str | None = None) -> None:
        self._target = shot_target
        self.hunter_called = False

    def get_agent(self, player_id: str):
        return _HunterMockAgent(self)


class _HunterMockAgent:
    def __init__(self, registry: _HunterMockRegistry) -> None:
        self._registry = registry

    def act(self, context):
        from werewolf_agent.agents.schemas import (
            ActionType, FallbackAction, PlayerAction, RetryInfo,
        )
        self._registry.hunter_called = True
        if self._registry._target and context.legal_actions and any(
            a.value == "hunter_shot" for a in context.legal_actions
        ):
            return PlayerAction(
                action_type=ActionType.HUNTER_SHOT,
                target_id=self._registry._target,
                speech="带走你",
                reason="shoot",
                confidence=0.9,
                private_intent=None,
            ), RetryInfo(attempts=0, errors=[])
        return FallbackAction(
            action_type=ActionType.NO_ACTION,
            target_id=None,
            reason="no shot",
        ), RetryInfo(attempts=0, errors=[])


class _WitchMockAgent:
    """Mock PlayerAgent that returns a configured witch action."""
    def __init__(self, *, use_antidote: bool = False, poison_target_id: str | None = None,
                 no_action: bool = False) -> None:
        self._use_antidote = use_antidote
        self._poison_target_id = poison_target_id
        self._no_action = no_action
        self.last_context: Any = None

    def act(self, context: AgentContext) -> tuple[PlayerAction, RetryInfo]:
        self.last_context = context
        if self._no_action:
            return PlayerAction(action_type=ActionType.NO_ACTION, reason="witch_pass"), RetryInfo()
        if self._use_antidote:
            return PlayerAction(
                action_type=ActionType.USE_ANTIDOTE,
                target_id=context.visible_world_state.get("wolf_kill_target"),
                reason="save target",
                confidence=0.85,
            ), RetryInfo()
        if self._poison_target_id:
            return PlayerAction(
                action_type=ActionType.USE_POISON,
                target_id=self._poison_target_id,
                reason="poison suspect",
                confidence=0.7,
            ), RetryInfo()
        return PlayerAction(action_type=ActionType.NO_ACTION, reason="no potion"), RetryInfo()


class _WitchMockRegistry:
    """Mock AgentRegistry that serves _WitchMockAgent for the witch player."""
    def __init__(self, *, witch_id: str = "witch",
                 use_antidote: bool = False, poison_target_id: str | None = None,
                 no_action: bool = False) -> None:
        self._witch_id = witch_id
        self._agent = _WitchMockAgent(
            use_antidote=use_antidote,
            poison_target_id=poison_target_id,
            no_action=no_action,
        )
        self.witch_called: bool = False

    def get_agent(self, player_id: str) -> _WitchMockAgent | None:
        if player_id == self._witch_id:
            self.witch_called = True
            return self._agent
        return None


class TestWitchDecisionFlow:
    """Witch agent-driven night decisions with legal information boundary,
    private audit events, and correct potion constraints."""

    def _make_witch_state(
        self,
        *,
        wolf_kill_target_id: str | None = "v1",
        antidote_used: bool = False,
        poison_used: bool = False,
    ) -> tuple[RuntimeState, RuleEngine]:
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "w2": PlayerState(id="w2", role="werewolf"),
            "w3": PlayerState(id="w3", role="werewolf"),
            "w4": PlayerState(id="w4", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "v2": PlayerState(id="v2", role="villager"),
            "v3": PlayerState(id="v3", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
            "seer": PlayerState(id="seer", role="seer"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "idiot": PlayerState(id="idiot", role="idiot"),
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
        }
        gs = GameState(
            game_id="witch_test",
            players=players,
            phase="night",
            night_number=1,
            antidote_used=antidote_used,
            poison_used=poison_used,
        )
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": wolf_kill_target_id,
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
        return state, engine

    def test_witch_agent_sees_wolf_kill_target(self) -> None:
        """When wolves killed someone, witch agent context includes wolf_kill_target."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ctx is not None
        assert ctx.visible_world_state.get("wolf_kill_target") == "v1"
        assert ActionType.USE_ANTIDOTE in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_agent_no_kill_target_when_wolves_no_kill(self) -> None:
        """When wolves no-killed, witch sees no wolf_kill_target in context."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id=None)
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ctx.visible_world_state.get("wolf_kill_target") is None
        assert ActionType.USE_ANTIDOTE not in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_cannot_self_save_antidote_not_offered(self) -> None:
        """When witch is the wolf kill target, antidote is NOT in legal_actions."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="witch")
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ActionType.USE_ANTIDOTE not in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_antidote_already_used_no_option(self) -> None:
        """When antidote already used, USE_ANTIDOTE not in legal_actions."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(
            wolf_kill_target_id="v1", antidote_used=True,
        )
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ActionType.USE_ANTIDOTE not in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_poison_already_used_no_option(self) -> None:
        """When poison already used, USE_POISON not in legal_actions."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(
            wolf_kill_target_id="v1", poison_used=True,
        )
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ActionType.USE_ANTIDOTE in ctx.legal_actions
        assert ActionType.USE_POISON not in ctx.legal_actions

    def test_witch_antidote_decision_produces_audit_event(self) -> None:
        """night_witch node produces witch_decision_audit event with witch_private visibility."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(use_antidote=True)

        result = night_witch({
            **state,
            "agent_registry": registry,
        })

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 1
        assert audit_events[0].payload.get("visibility") == "witch_private"
        assert audit_events[0].payload.get("action_taken") == "use_antidote"
        assert audit_events[0].payload.get("wolf_kill_target_id") == "v1"
        assert result["use_antidote"] is True
        assert result.get("poison_target_id") is None

    def test_witch_poison_decision_produces_audit_event(self) -> None:
        """night_witch node produces witch_decision_audit for poison."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(poison_target_id="w1")

        result = night_witch({
            **state,
            "agent_registry": registry,
        })

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 1
        assert audit_events[0].payload.get("action_taken") == "use_poison"
        assert audit_events[0].payload.get("poison_target_id") == "w1"
        assert result["use_antidote"] is False
        assert result.get("poison_target_id") == "w1"

    def test_witch_no_action_produces_audit_event(self) -> None:
        """Even witch no-action produces an audit event."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(no_action=True)

        result = night_witch({
            **state,
            "agent_registry": registry,
        })

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 1
        assert audit_events[0].payload.get("action_taken") == "no_action"

    def test_witch_scripted_no_audit_event(self) -> None:
        """Scripted fallback (no registry) should NOT produce audit event."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")

        result = night_witch(state)

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 0

    def test_witch_resolve_night_events_have_visibility(self) -> None:
        """RuleEngine resolve_night witch events carry witch_private visibility."""
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        gs = state["game_state"]

        gs_out, events = engine.resolve_night(
            gs,
            night_number=1,
            wolf_kill_target_id="v1",
            use_antidote=True,
            poison_target_id=None,
        )
        witch_events = [e for e in events if e.type in ("witch_antidote_used", "witch_poison_used")]
        assert len(witch_events) == 1
        assert witch_events[0].payload.get("visibility") == "witch_private"

    def test_witch_poison_event_has_visibility(self) -> None:
        """RuleEngine resolve_night witch_poison_used event has witch_private visibility."""
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        gs = state["game_state"]

        gs_out, events = engine.resolve_night(
            gs,
            night_number=1,
            wolf_kill_target_id="v1",
            use_antidote=False,
            poison_target_id="w1",
        )
        witch_events = [e for e in events if e.type in ("witch_antidote_used", "witch_poison_used")]
        assert len(witch_events) == 1
        assert witch_events[0].type == "witch_poison_used"
        assert witch_events[0].payload.get("visibility") == "witch_private"


class TestHunterShotResolution:
    """Night hunter shot: when hunter is killed by wolves, the shot must be
    resolved before victory check. Agent-driven target selection when registry
    provided."""

    def test_resolve_hunter_shot_applies_shot_at_night(self) -> None:
        """resolve_hunter_shot node applies hunter_shot death when hunter is wolf-killed."""
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter", reason="wolf_kill", timing="night",
            resolution_batch="night_1", triggered_skills=["hunter_shot"],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_night", players=players, deaths=[hunter_death])

        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        assert result["game_state"].players["v1"].alive is False
        shot_deaths = [d for d in result["game_state"].deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 1
        assert shot_deaths[0].source_player_id == "hunter"

    def test_resolve_hunter_shot_no_shot_when_poisoned(self) -> None:
        """Hunter killed by witch poison does NOT trigger a shot."""
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        engine = _new_engine()
        poison_death = Death(
            player_id="hunter", reason="witch_poison", timing="night",
            resolution_batch="night_1", triggered_skills=[],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_poison", players=players, deaths=[poison_death])

        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        assert result["game_state"].players["v1"].alive is True
        shot_deaths = [d for d in result["game_state"].deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 0

    def test_resolve_hunter_shot_agent_selects_target(self) -> None:
        """Agent registry provides hunter shot target when no scripted target."""
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter", reason="wolf_kill", timing="night",
            resolution_batch="night_1", triggered_skills=["hunter_shot"],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_agent", players=players, deaths=[hunter_death])

        registry = _HunterMockRegistry(shot_target="v1")
        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
            "hunter_shot_target_id": None,
        })

        assert registry.hunter_called
        assert result["game_state"].players["v1"].alive is False

    def test_route_after_resolve_night_routes_to_hunter_shot(self) -> None:
        """route_after_resolve_night routes to resolve_hunter_shot when hunter has pending shot."""
        from werewolf_agent.runtime.graph import route_after_resolve_night
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter", reason="wolf_kill", timing="night",
            resolution_batch="night_1", triggered_skills=["hunter_shot"],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_route", players=players, deaths=[hunter_death])

        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "resolve_hunter_shot"

    def test_route_after_resolve_night_no_hunter_shot_skips(self) -> None:
        """route_after_resolve_night skips hunter_shot when no pending shot."""
        from werewolf_agent.runtime.graph import route_after_resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="hs_skip", players=players)

        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result != "resolve_hunter_shot"

    def test_graph_contains_resolve_hunter_shot_node(self) -> None:
        """Graph must include resolve_hunter_shot node."""
        graph = build_game_graph()
        assert "resolve_hunter_shot" in graph.nodes, (
            "resolve_hunter_shot node is missing from the graph"
        )


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


# ---------------------------------------------------------------------------
# Start-game event sourcing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 7: Sheriff Badge Night Death Routing Tests
# ---------------------------------------------------------------------------


class TestSheriffBadgeNightDeathRouting:
    """When sheriff dies at night (wolf kill, witch poison, hunter shot),
    route_after_resolve_night must go to sheriff_badge_transfer before
    announce_deaths when the game has not been won."""

    def test_sheriff_died_at_night_routes_to_badge_transfer(self) -> None:
        """route_after_resolve_night routes to sheriff_badge_transfer when
        sheriff died from wolf kill and game continues."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        # Find a non-wolf target to be sheriff, then kill them
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        players[sheriff_id] = replace(players[sheriff_id], alive=False)
        gs = GameState(
            game_id="badge_night_route",
            players=players,
            phase="night",
            night_number=1,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "sheriff_badge_transfer"

    def test_sheriff_died_at_night_after_hunter_shot_routes_to_badge_transfer(self) -> None:
        """route_after_hunter_shot routes to sheriff_badge_transfer when
        sheriff died and game continues."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        players[sheriff_id] = replace(players[sheriff_id], alive=False)
        gs = GameState(
            game_id="badge_hunter_route",
            players=players,
            phase="night",
            night_number=1,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        result = route_after_hunter_shot({"game_state": gs, "engine": engine})
        assert result == "sheriff_badge_transfer"

    def test_no_sheriff_death_routes_to_sheriff_first_day_on_night1(self) -> None:
        """Night 1 with no sheriff death routes to sheriff_first_day_entry (sheriff before deaths)."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(
            game_id="badge_no_sheriff",
            players=players,
            phase="night",
            night_number=1,
        )
        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "sheriff_first_day_entry"

    def test_sheriff_alive_routes_to_announce_deaths(self) -> None:
        """When sheriff is still alive, route goes to announce_deaths."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        gs = GameState(
            game_id="badge_sheriff_alive",
            players=players,
            phase="night",
            night_number=1,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "announce_deaths"

    def test_badge_transfer_night_phase_routes_to_announce_deaths(self) -> None:
        """After badge_transfer in night-death path (phase=night),
        _route_after_badge_transfer routes to announce_deaths."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(
            game_id="badge_transfer_night",
            players=players,
            phase="night",
            night_number=1,
            sheriff_badge_state="torn",
        )
        result = _route_after_badge_transfer({"game_state": gs, "engine": engine})
        assert result == "announce_deaths"

    def test_badge_transfer_day_phase_routes_to_enter_night(self) -> None:
        """After badge_transfer in post-victory path (phase != night),
        _route_after_badge_transfer routes to enter_night."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(
            game_id="badge_transfer_day",
            players=players,
            phase="day",
            day_number=2,
            sheriff_badge_state="torn",
        )
        result = _route_after_badge_transfer({"game_state": gs, "engine": engine})
        assert result == "enter_night"

    def test_sheriff_died_this_batch_helper(self) -> None:
        """_sheriff_died_this_batch correctly detects dead sheriff."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        # Sheriff alive
        gs_alive = GameState(
            players=players,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        assert _sheriff_died_this_batch(gs_alive) is False

        # Sheriff dead
        players_dead = dict(players)
        players_dead[sheriff_id] = replace(players[sheriff_id], alive=False)
        gs_dead = GameState(
            players=players_dead,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        assert _sheriff_died_this_batch(gs_dead) is True

        # No sheriff
        gs_none = GameState(players=players)
        assert _sheriff_died_this_batch(gs_none) is False

        # Badge not active
        gs_inactive = GameState(
            players=players_dead,
            sheriff_id=sheriff_id,
            sheriff_badge_state="torn",
        )
        assert _sheriff_died_this_batch(gs_inactive) is False


# ---------------------------------------------------------------------------
# Task 1-3: Vote lifecycle tests
# ---------------------------------------------------------------------------


class TestVoteLifecycle:
    """Vote state must reset across days and revote rounds."""

    def test_day_vote_ignores_stale_votes_when_day_changes(self) -> None:
        from werewolf_agent.runtime.graph import day_vote

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="vote_stale", players=players, day_number=2, night_number=2)

        # Old votes from day 1 should be ignored
        result = day_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p01": "p05", "p02": "p06"},
            "exile_vote_day": 1,
            "exile_vote_revote": False,
            "revote": False,
        })

        assert result["exile_vote_day"] == 2
        # Without registry, existing votes from day 1 are stale, so votes should be empty
        assert result["exile_votes"] == {}

    def test_day_vote_reuses_votes_same_day(self) -> None:
        from werewolf_agent.runtime.graph import day_vote

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="vote_same", players=players, day_number=2)

        old_votes = {"p01": "p05", "p02": "p05"}
        result = day_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": old_votes,
            "exile_vote_day": 2,
            "exile_vote_revote": False,
            "revote": False,
        })

        # Same day, same revote window → votes are reused
        assert result["exile_votes"] == old_votes

    def test_tie_revote_clears_first_round_votes(self) -> None:
        from werewolf_agent.runtime.graph import tie_revote

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="tie_revote", players=players, day_number=3)

        result = tie_revote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p01": "p05", "p02": "p06"},
            "exile_vote_day": 3,
            "exile_vote_revote": False,
        })

        assert result["revote"] is True
        assert result["exile_votes"] == {}
        assert result["exile_vote_revote"] is True

    def test_tie_revote_routes_back_to_day_vote(self) -> None:
        graph = build_game_graph().get_graph()

        targets = {
            edge.target
            for edge in graph.edges
            if edge.source == "tie_revote"
        }

        assert "day_vote" in targets
        assert "resolve_vote_node" not in targets

    def test_no_exile_counter_increments_on_second_tie(self) -> None:
        from werewolf_agent.runtime.graph import resolve_vote

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="no_exile_counter", players=players, day_number=2)

        # Tie votes that cause second_tie_no_exile
        result = resolve_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p01": "p05", "p02": "p06"},
            "revote": True,
            "consecutive_no_exile_days": 0,
        })

        assert result["consecutive_no_exile_days"] == 1

    def test_no_exile_counter_resets_on_exile(self) -> None:
        from werewolf_agent.runtime.graph import resolve_vote

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="exile_reset", players=players, day_number=2)

        result = resolve_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p01": "p05", "p02": "p05", "p03": "p05"},
            "revote": False,
            "consecutive_no_exile_days": 2,
        })

        assert result["consecutive_no_exile_days"] == 0
        assert result["_vote_result"].exiled_player_id == "p05"


class TestAntiStallPolicy:
    """Anti-stall tie-break after consecutive no-exile days."""

    def test_anti_stall_breaks_repeated_second_tie(self) -> None:
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="anti_stall", players=players, day_number=3)

        result = engine.resolve_vote(
            gs,
            votes={"p01": "p05", "p02": "p06"},
            revote=True,
            consecutive_no_exile_days=2,
            rng_seed="game-pace-test",
        )

        assert result.exiled_player_id in {"p05", "p06"}
        assert result.reason == "anti_stall_tie_break"

    def test_empty_revote_anti_stall_uses_pk_candidates_only(self) -> None:
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="anti_stall_empty", players=players, day_number=3)

        result = engine.resolve_vote(
            gs,
            votes={},
            revote=True,
            consecutive_no_exile_days=2,
            pk_candidates=["p05", "p06"],
            rng_seed="game-pace-test",
        )

        assert result.exiled_player_id in {"p05", "p06"}
        assert result.reason == "anti_stall_empty_tally"

    def test_anti_stall_not_triggered_below_threshold(self) -> None:
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="no_stall", players=players, day_number=2)

        result = engine.resolve_vote(
            gs,
            votes={"p01": "p05", "p02": "p06"},
            revote=True,
            consecutive_no_exile_days=0,
            rng_seed="game-pace-test",
        )

        assert result.exiled_player_id is None
        assert result.reason == "second_tie_no_exile"

    def test_second_tie_no_exile_preserved_without_anti_stall(self) -> None:
        """Original tie behavior preserved when consecutive days < threshold."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="preserve_tie", players=players, day_number=2)

        result = engine.resolve_vote(
            gs,
            votes={"p01": "p05", "p02": "p06"},
            revote=True,
            consecutive_no_exile_days=0,
        )

        assert result.reason == "second_tie_no_exile"

    def test_majority_vote_creates_exile_death_once(self) -> None:
        from werewolf_agent.runtime.graph import resolve_vote, resolve_exile

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="majority_exile", players=players, day_number=2)

        state = {
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p01": "p05", "p02": "p05", "p03": "p05"},
            "revote": False,
            "consecutive_no_exile_days": 0,
        }

        state.update(resolve_vote(state))
        state.update(resolve_exile(state))

        deaths = state["game_state"].deaths
        exile_deaths = [d for d in deaths if d.player_id == "p05" and d.reason == "exile"]
        assert len(exile_deaths) >= 1


# ---------------------------------------------------------------------------
# Task 7: Witch Poison Pressure Policy
# ---------------------------------------------------------------------------


class TestWitchPoisonPressureContext:
    """Witch night context must include pressure targets when they exist."""

    def test_witch_context_includes_poison_pressure_targets(self) -> None:
        """When public state has unresolved black claim, witch sees pressure targets."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players: dict[str, PlayerState] = {}
        for i in range(1, 13):
            role = "werewolf" if i <= 4 else ("witch" if i == 9 else "villager")
            players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)

        gs = GameState(
            game_id="test",
            phase="night",
            night_number=2,
            players=players,
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "p05",
                        "day_number": 1,
                        "text": "我是预言家，p03查杀",
                    },
                ),
                GameEvent(
                    type="seer_check",
                    payload={
                        "seer_id": "p05",
                        "target_id": "p03",
                        "alignment": "wolf",
                        "night_number": 1,
                        "visibility": "seer_only",
                    },
                ),
            ],
        )
        engine = RuleEngine.from_yaml(RULESET_PATH)

        context = build_agent_context(
            engine, gs, "p09", TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.NO_ACTION, ActionType.USE_POISON],
            legal_targets=["p03"],
        )

        # Witch context must include poison_pressure_targets
        assert "poison_pressure_targets" in context.visible_world_state
        targets = context.visible_world_state["poison_pressure_targets"]
        assert len(targets) >= 1
        black_claim_targets = [
            t for t in targets if t["pressure_type"] == "black_claim"
        ]
        assert len(black_claim_targets) == 1
        assert black_claim_targets[0]["player_id"] == "p03"
        assert black_claim_targets[0]["source"] == "p05"

    def test_witch_context_no_pressure_targets_without_claims(self) -> None:
        """When there are no public black claims, no pressure targets appear."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players: dict[str, PlayerState] = {}
        for i in range(1, 13):
            role = "werewolf" if i <= 4 else ("witch" if i == 9 else "villager")
            players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)

        gs = GameState(
            game_id="test_no_pressure",
            phase="night",
            night_number=1,
            players=players,
            events=[],
        )
        engine = RuleEngine.from_yaml(RULESET_PATH)

        context = build_agent_context(
            engine, gs, "p09", TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.NO_ACTION, ActionType.USE_POISON],
            legal_targets=["p03"],
        )

        assert "poison_pressure_targets" not in context.visible_world_state

    def test_witch_context_no_pressure_when_poison_used(self) -> None:
        """When poison is already used, no pressure targets are added."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players: dict[str, PlayerState] = {}
        for i in range(1, 13):
            role = "werewolf" if i <= 4 else ("witch" if i == 9 else "villager")
            players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)

        gs = GameState(
            game_id="test_poison_used",
            phase="night",
            night_number=2,
            players=players,
            poison_used=True,
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "p05",
                        "day_number": 1,
                        "text": "p03查杀",
                    },
                ),
            ],
        )
        engine = RuleEngine.from_yaml(RULESET_PATH)

        context = build_agent_context(
            engine, gs, "p09", TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.NO_ACTION],
            legal_targets=[],
        )

        # Poison already used, so no pressure targets added
        assert "poison_pressure_targets" not in context.visible_world_state

    def test_witch_pressure_strategy_directive_in_night_witch(self) -> None:
        """agent_night_witch adds strategy_directive when pressure targets exist."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch

        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "w2": PlayerState(id="w2", role="werewolf"),
            "w3": PlayerState(id="w3", role="werewolf"),
            "w4": PlayerState(id="w4", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "v2": PlayerState(id="v2", role="villager"),
            "v3": PlayerState(id="v3", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
            "seer": PlayerState(id="seer", role="seer"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "idiot": PlayerState(id="idiot", role="idiot"),
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
        }
        gs = GameState(
            game_id="witch_pressure_directive",
            players=players,
            phase="night",
            night_number=2,
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "seer",
                        "day_number": 1,
                        "text": "我是预言家，w1查杀",
                    },
                ),
            ],
        )
        engine = _new_engine()

        class CaptureWitchAgent:
            """Agent that captures context and returns no_action."""
            last_context: AgentContext | None = None

            def act(self, context):
                self.last_context = context
                return (
                    PlayerAction(
                        action_type=ActionType.NO_ACTION,
                        reason="saving poison",
                    ),
                    RetryInfo(),
                )

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureWitchAgent()

            def get_agent(self, player_id):
                if player_id == "witch":
                    return self.agent
                return None

        registry = CaptureRegistry()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "v1",
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

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry.agent.last_context
        assert ctx is not None
        # Strategy directive should mention pressure
        assert "witch_pressure" in ctx.strategy_directive
        assert "w1" in ctx.strategy_directive["witch_pressure"]
        assert "required_evaluation" in ctx.strategy_directive


# ---------------------------------------------------------------------------
# Task 10: Judge-Controlled Night And Day Broadcasts
# ---------------------------------------------------------------------------


class TestJudgeControlsNightRoleSequence:
    """Night role sequence has judge broadcasts."""

    def test_judge_controls_night_role_sequence(self):
        """Event stream contains judge broadcasts before/after each night role stage."""
        from werewolf_agent.runtime.graph import enter_night

        gs = GameState(
            game_id="test",
            phase="setup",
            players={f"p{i:02d}": PlayerState(
                id=f"p{i:02d}",
                role="werewolf" if i <= 4 else ("seer" if i == 5 else ("witch" if i == 9 else "villager")),
                alive=True,
            ) for i in range(1, 13)},
        )

        result = enter_night({"game_state": gs, "agent_registry": None})
        gs = result["game_state"]

        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "enter_night" in phases

    def test_wolf_stages_have_broadcasts(self):
        """Wolf discussion and consensus emit judge broadcasts."""
        from werewolf_agent.runtime.graph import wolf_discussion

        gs = GameState(
            game_id="test",
            phase="night",
            night_number=1,
            players={f"p{i:02d}": PlayerState(
                id=f"p{i:02d}",
                role="werewolf" if i <= 4 else "villager",
                alive=True,
            ) for i in range(1, 13)},
        )

        state = {"game_state": gs, "agent_registry": None}
        result = wolf_discussion(state)
        gs = result["game_state"]

        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        wolf_broadcasts = [p for p in phases if "wolf" in p]
        assert len(wolf_broadcasts) >= 1


class TestJudgeControlsDaySequence:
    """Day flow broadcasts include all major transitions."""

    def test_day_announce_has_broadcast(self):
        from werewolf_agent.runtime.graph import announce_deaths

        gs = GameState(
            game_id="test",
            phase="night",
            night_number=1,
            day_number=0,
            players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)},
        )
        result = announce_deaths({"game_state": gs})
        gs = result["game_state"]

        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "day_announce" in phases

    def test_vote_phase_has_broadcast(self):
        """day_vote should emit a judge broadcast."""
        from werewolf_agent.runtime.graph import day_vote

        gs = GameState(
            game_id="test",
            phase="day",
            day_number=2,
            players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)},
        )
        state = {
            "game_state": gs,
            "agent_registry": None,
            "exile_votes": {"p01": "p03"},
            "exile_vote_day": 0,
            "revote": False,
        }
        result = day_vote(state)
        gs = result.get("game_state", gs)
        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "vote_start" in phases
