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


def test_wolf_consensus_timeout_defaults_to_no_kill_event() -> None:
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="wolf_timeout", players=players, night_number=1)

    result = wolf_consensus({"game_state": gs, "engine": engine})

    assert result["wolf_kill_target_id"] is None
    event = _last_non_broadcast_event(result["game_state"])
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
    event = _last_non_broadcast_event(result["game_state"])
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
    event = _last_non_broadcast_event(result["game_state"])
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
    event = _last_non_broadcast_event(result["game_state"])
    assert event.type == "wolf_no_kill_timeout"
    assert event.payload["reason"] == "timer_expired"


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
    from werewolf_agent.runtime.nodes import day as day_mod

    gs = GameState(game_id="speech_trace_private", day_number=1)
    private_trace = {
        "raw_text": '{"private_intent":{"true_role":"werewolf"}}',
        "parsed_action": {
            "action_type": "speech",
            "private_intent": {"true_role": "werewolf"},
        },
        "final_action_type": "speech",
    }

    def fake_dispatch_agent(state, fn, *extra_args, **kwargs):
        return {"speech_text": "公开发言", "action_trace": private_trace}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(day_mod, "_dispatch_agent", fake_dispatch_agent)

    result = day_mod.free_discussion({
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


def test_sheriff_vote_ignores_candidates_and_withdrew_voters() -> None:
    from werewolf_agent.runtime.graph import sheriff_vote

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
    }
    gs = GameState(
        game_id="sheriff_vote_eligibility",
        players=players,
        day_number=1,
        sheriff_candidates=["p01", "p02"],
    )

    result = sheriff_vote({
        "game_state": gs,
        "engine": _new_engine(),
        "sheriff_withdrawing": ["p03"],
        "sheriff_votes": {
            "p01": "p01",  # candidate cannot vote
            "p03": "p01",  # withdrew candidate still cannot vote
            "p04": "p02",  # only valid off-sheriff vote
        },
    })

    assert result["game_state"].sheriff_id == "p02"


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


def test_resolve_vote_records_sheriff_weighted_tally() -> None:
    from werewolf_agent.runtime.graph import resolve_vote

    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="villager"),
        "p03": PlayerState(id="p03", role="werewolf"),
    }
    gs = GameState(
        game_id="sheriff_weighted_vote",
        players=players,
        sheriff_id="p01",
        sheriff_badge_state="active",
        day_number=2,
    )

    result = resolve_vote({
        "game_state": gs,
        "engine": _new_engine(),
        "exile_votes": {"p01": "p03", "p02": "p02"},
        "revote": False,
    })

    vote_event = next(event for event in result["game_state"].events if event.type == "vote_resolved")
    assert vote_event.payload["sheriff_id"] == "p01"
    assert vote_event.payload["sheriff_vote_weight"] == 1.5
    assert vote_event.payload["weighted_tally"] == {"p03": 1.5, "p02": 1.0}
    assert vote_event.payload["vote_weights"] == {"p01": 1.5, "p02": 1.0}


def test_resolve_vote_first_tie_emits_pk_broadcast() -> None:
    from werewolf_agent.runtime.graph import resolve_vote

    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="villager"),
        "p03": PlayerState(id="p03", role="werewolf"),
        "p04": PlayerState(id="p04", role="villager"),
    }
    gs = GameState(game_id="first_tie_pk", players=players, day_number=2)

    result = resolve_vote({
        "game_state": gs,
        "engine": _new_engine(),
        "exile_votes": {"p01": "p03", "p02": "p04"},
        "revote": False,
    })

    broadcasts = [
        event for event in result["game_state"].events
        if event.type == "judge_broadcast" and event.payload.get("phase") == "vote_tie_pk"
    ]
    assert broadcasts
    assert result["pk_candidates"] == ["p03", "p04"]


def test_vote_action_trace_audit_exposes_structured_private_vote_thought_to_moderator_only(capsys) -> None:
    from werewolf_agent.runtime.graph import resolve_vote
    from werewolf_agent.runtime.public_ledger import build_public_ledger

    engine = _new_engine()
    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="werewolf"),
    }
    gs = GameState(game_id="vote_private_thought", players=players, day_number=1)
    private_trace = {
        "parsed_action": {
            "reason": "公开理由：跟随查杀",
            "private_reason": "心里想：p02的发言像倒钩狼，先投他试压力",
            "standing_with_seer": "p03",
            "suspect_reason": "p02警上站边摇摆，且投票理由跟风",
            "not_voting_reason": "p04虽然发言短，但没有和悍跳线绑定",
            "private_intent": {"true_role": "villager"},
        },
    }

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

    assert audit_event.payload["visibility"] == "moderator_only"
    assert audit_event.payload["day_number"] == 1
    assert audit_event.payload["private_vote_thought"] == {
        "target": "p02",
        "public_reason": "公开理由：跟随查杀",
        "standing_with_seer": "p03",
        "suspect_reason": "p02警上站边摇摆，且投票理由跟风",
        "not_voting_reason": "p04虽然发言短，但没有和悍跳线绑定",
        "private_reason": "心里想：p02的发言像倒钩狼，先投他试压力",
    }
    assert audit_event.payload["vote_target"] == "p02"
    assert "private_vote_thought" not in vote_event.payload
    assert "心里想" not in str(vote_event.payload)
    assert "心里想" not in str(build_public_ledger(result["game_state"]))


def test_first_night_wolf_discussion_runs_three_rounds_and_builds_team_plan(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import night as night_mod

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

    def fake_dispatch_agent(_state, _fn, *_extra_args, **_kwargs):
        wolf_id = _extra_args[0]
        calls.append((wolf_id, _state.get("wolf_discussion_round")))
        return {"speech_text": f"{wolf_id} round {_state.get('wolf_discussion_round')}"}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch_agent)

    result = night_mod.wolf_discussion({
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
        "public_story",
    ):
        assert plan[key]
    assert plan["night_kill_primary"] is None
    assert plan["night_kill_backup"] is None
    assert plan["day_push_target"] is None
    assert plan["evidence_quality"] == "none"

    assignments = [plan["fake_seer"], plan["pusher"], plan["hooker"], plan["deep_cover"]]
    assert sorted(assignments) == ["w1", "w2", "w3", "w4"]


def test_later_night_wolf_discussion_runs_two_rounds_and_revises_plan(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import night as night_mod

    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="villager"),
    }
    gs = GameState(game_id="wolf_plan_later", players=players, night_number=2, phase="night")

    def fake_dispatch_agent(_state, _fn, *_extra_args, **_kwargs):
        return {"speech_text": "revise plan"}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch_agent)

    result = night_mod.wolf_discussion({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
        "wolf_team_plan": {"fake_seer": "w1", "pusher": "w2"},
    })

    round_events = [event for event in result["game_state"].events if event.type == "wolf_discussion"]
    assert len(round_events) == 4
    assert {event.payload["round"] for event in round_events} == {1, 2}
    assert result["wolf_team_plan"]["night_number"] == 2


def test_wolf_discussion_drops_stale_targets_without_current_discussion_evidence(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import night as night_mod

    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="villager"),
    }
    gs = GameState(game_id="wolf_plan_stale", players=players, night_number=3, phase="night")

    def fake_dispatch_agent(_state, _fn, *_extra_args, **_kwargs):
        return {"speech_text": "今晚先重新听意见，暂时不点明确刀口。"}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch_agent)

    result = night_mod.wolf_discussion({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
        "wolf_team_plan": {
            "night_kill_primary": "v1",
            "night_kill_backup": "v2",
            "day_push_target": "v1",
            "evidence_quality": "strong",
            "evidence_from_discussion": [{"target": "v1", "reason": "old night"}],
            "fake_seer": "w1",
            "pusher": "w2",
        },
    })

    plan = result["wolf_team_plan"]
    assert plan["night_kill_primary"] is None
    assert plan["night_kill_backup"] is None
    assert plan["day_push_target"] is None
    assert plan["evidence_quality"] == "none"


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
            "evidence_quality": "strong",
            "evidence_from_discussion": [{"target": "v2"}],
        },
    })

    assert result["wolf_kill_target_id"] == "v2"
    event = _last_non_broadcast_event(result["game_state"])
    assert event.type == "wolf_kill_selected"
    assert event.payload["target_id"] == "v2"
    assert event.payload["reason"] == "wolf_team_plan"


def test_resolve_vote_records_vote_reasons_for_public_ledger() -> None:
    from werewolf_agent.runtime.graph import resolve_vote

    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="villager"),
        "p08": PlayerState(id="p08", role="werewolf"),
    }
    gs = GameState(game_id="vote_public_ledger", players=players, day_number=2)

    result = resolve_vote({
        "game_state": gs,
        "engine": _new_engine(),
        "exile_votes": {"p01": "p08", "p02": "p08"},
        "vote_action_traces": {
            "p01": {"parsed_action": {"reason": "跟预言家查杀", "private_intent": {"true_role": "villager"}}},
            "p02": {"reason": "票型跟随"},
        },
        "revote": False,
    })

    vote_event = [
        event for event in result["game_state"].events
        if event.type == "vote_resolved"
    ][0]

    assert vote_event.payload["day_number"] == 2
    assert vote_event.payload["votes"] == [
        {"voter": "p01", "target": "p08", "reason": "跟预言家查杀"},
        {"voter": "p02", "target": "p08", "reason": "票型跟随"},
    ]
    assert "private_intent" not in str(vote_event.payload)


def test_resolve_vote_uses_fallback_reason_for_public_ledger() -> None:
    from werewolf_agent.runtime.graph import resolve_vote

    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p08": PlayerState(id="p08", role="werewolf"),
    }
    gs = GameState(game_id="vote_fallback_reason", players=players, day_number=2)

    result = resolve_vote({
        "game_state": gs,
        "engine": _new_engine(),
        "exile_votes": {"p01": "p08"},
        "vote_action_traces": {
            "p01": {
                "parsed_action": None,
                "fallback_reason": "fallback: 结构化输出失败，按当前可见线索选择p08",
            },
        },
        "revote": False,
    })

    vote_event = [
        event for event in result["game_state"].events
        if event.type == "vote_resolved"
    ][0]

    assert vote_event.payload["votes"] == [
        {
            "voter": "p01",
            "target": "p08",
            "reason": "fallback: 结构化输出失败，按当前可见线索选择p08",
        },
    ]


def test_agent_day_vote_excludes_voter_from_legal_targets() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_day_vote

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    gs = GameState(game_id="vote_no_self_target", players=players, day_number=1)

    class Agent:
        def __init__(self) -> None:
            self.context = None

        def act(self, context):
            self.context = context
            return PlayerAction(
                action_type=ActionType.VOTE,
                target_id="p02",
                reason="p02 has the weakest public logic",
            ), RetryInfo()

    agent = Agent()

    class Registry:
        def get_agent(self, player_id):
            return agent

    result = agent_day_vote({"game_state": gs}, _new_engine(), Registry(), "p01")

    assert result["vote_target"] == "p02"
    assert agent.context.legal_targets == ["p02", "p03"]
    assert "p01" not in agent.context.legal_targets


def test_sheriff_speech_calls_candidate_agents_and_keeps_trace_private(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import sheriff as sheriff_mod

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

    def fake_dispatch_agent(*_args, **_kwargs):
        return {"speech_text": "我上警竞选警长。", "action_trace": private_trace}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(sheriff_mod, "_dispatch_agent", fake_dispatch_agent)

    result = sheriff_mod.sheriff_speech({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
    })

    events = result["game_state"].events
    speeches = [event for event in events if event.type == "sheriff_speech"]
    audits = [event for event in events if event.type == "action_trace_audit"]

    assert sorted(event.payload["speaker"] for event in speeches) == ["p01", "p02"]
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


def test_day_speech_requires_speech_action_from_agent() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_day_speech

    players = {"p01": PlayerState(id="p01", role="villager")}
    gs = GameState(game_id="day_speech_requires_speech", players=players, day_number=1, phase="day")

    class Agent:
        def __init__(self):
            self.context = None

        def act(self, context):
            self.context = context
            return PlayerAction(
                action_type=ActionType.SPEECH,
                speech="我是好人阵营。我怀疑p02，p02发言前后矛盾。我倾向投p02。",
                reason="按发言逻辑分析",
            ), RetryInfo()

    agent = Agent()

    class Registry:
        def get_agent(self, player_id):
            return agent

    result = agent_day_speech({"game_state": gs}, _new_engine(), Registry(), "p01")

    assert result["speech_text"]
    assert agent.context.legal_actions == [ActionType.SPEECH]


def test_announce_deaths_skips_increment_when_phase_is_day() -> None:
    """When phase is already 'day', announce_deaths does not re-increment."""
    from werewolf_agent.runtime.graph import announce_deaths

    gs = GameState(
        game_id="day_marker_reset",
        players={"p01": PlayerState(id="p01", role="villager")},
        phase="day",
        day_number=1,
        night_number=1,
    )

    result = announce_deaths({"game_state": gs})

    assert result["game_state"].day_number == 1


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
    assert route_self_destruct_check(result) == "summarize_positions"


def test_free_discussion_announces_speech_order_and_discussion_end() -> None:
    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="villager"),
    }
    gs = GameState(game_id="day_broadcasts", players=players, day_number=2, phase="day")

    first = free_discussion({
        "game_state": gs,
        "speech_order": ["p01", "p02"],
        "speech_index": 0,
        "speech_text": "first speech",
    })
    second = free_discussion({
        **first,
        "speech_text": "second speech",
    })

    broadcasts = [
        e.payload for e in second["game_state"].events
        if e.type == "judge_broadcast"
    ]
    phases = [payload["phase"] for payload in broadcasts]

    assert "discussion_start" in phases
    assert "speech_order" in phases
    assert phases.count("speaker_turn") == 2
    assert phases[-1] == "discussion_end"
    speech_order = next(payload for payload in broadcasts if payload["phase"] == "speech_order")
    assert speech_order["speech_order"] == ["p01", "p02"]


def test_day_vote_announces_vote_collection_and_end() -> None:
    from werewolf_agent.runtime.graph import day_vote

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = GameState(game_id="vote_broadcasts", players=players, day_number=2, phase="day")

    result = day_vote({
        "game_state": gs,
        "agent_registry": None,
        "exile_votes": {"p01": "p02"},
        "exile_vote_day": 2,
        "exile_vote_revote": False,
        "revote": False,
    })

    phases = [
        e.payload.get("phase")
        for e in result["game_state"].events
        if e.type == "judge_broadcast"
    ]

    assert phases == ["vote_start", "vote_collect", "vote_end", "vote_result"]


def test_night_death_last_words_has_public_broadcast() -> None:
    from werewolf_agent.runtime.graph import night_death_last_words

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=False),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = GameState(
        game_id="night_last_words_broadcast",
        players=players,
        day_number=2,
        night_number=2,
        deaths=[Death(
            player_id="p01",
            reason="wolf_kill",
            timing="night",
            resolution_batch="night_2",
            can_leave_last_words=True,
        )],
    )

    result = night_death_last_words({"game_state": gs, "engine": _new_engine()})

    broadcasts = [
        e.payload for e in result["game_state"].events
        if e.type == "judge_broadcast"
    ]
    assert broadcasts[-1]["phase"] == "night_death_last_words"
    assert broadcasts[-1]["players"] == ["p01"]
    assert broadcasts[-1]["visibility"] == "public"


def test_night_death_last_words_broadcasts_skip_when_no_eligible_players() -> None:
    from werewolf_agent.runtime.graph import night_death_last_words

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=False),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = GameState(
        game_id="night_last_words_skip_broadcast",
        players=players,
        day_number=3,
        night_number=2,
        deaths=[Death(
            player_id="p01",
            reason="wolf_kill",
            timing="night",
            resolution_batch="night_2",
            can_leave_last_words=False,
        )],
    )

    result = night_death_last_words({"game_state": gs, "engine": _new_engine()})

    broadcasts = [
        e.payload for e in result["game_state"].events
        if e.type == "judge_broadcast"
    ]
    assert broadcasts[-1]["phase"] == "night_death_last_words"
    assert broadcasts[-1]["players"] == []


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


def test_route_after_announce_night1_goes_to_free_discussion() -> None:
    """N1 sheriff election now runs BEFORE announce_deaths, so after announce
    the router goes straight to free_discussion."""
    from werewolf_agent.runtime.graph import route_after_announce
    engine = _new_engine()
    gs = GameState(day_number=1, night_number=1)
    assert route_after_announce({"game_state": gs, "engine": engine}) == "free_discussion"


def test_route_after_announce_day2_discussion() -> None:
    from werewolf_agent.runtime.graph import route_after_announce
    gs = GameState(day_number=2, night_number=2)
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
    }) == "summarize_positions"


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
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/rules/test_rule_engine_v1.py", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Phase 1 tests failed: {result.stdout[:500]}"


# ---------------------------------------------------------------------------
# Task 1: Runtime Gap Tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Backward-compat re-imports: classes moved to domain-specific test files
# ---------------------------------------------------------------------------

# test_night_flow.py
from tests.runtime.test_night_flow import (
    TestNightHunterIdiotStatusNode,
    TestSeerNightResolution,
)

# test_wolf_flow.py
from tests.runtime.test_wolf_flow import (
    TestWolfDiscussionLoop,
    TestWolfFallbackVoteNoTeammate,
    TestWolfPlanDedup,
)

# test_witch_flow.py
from tests.runtime.test_witch_flow import (
    TestWitchDecisionFlow,
    TestWitchPoisonPressureContext,
)

# test_hunter_flow.py
from tests.runtime.test_hunter_flow import (
    TestHunterShotTiming,
    TestHunterShotOrdering,
    TestHunterShotResolution,
    TestHunterShotPublicEvent,
)

# test_sheriff_flow.py
from tests.runtime.test_sheriff_flow import (
    TestSheriffBadgeAfterNightDeath,
    TestSheriffBadgeNightDeathRouting,
    TestSheriffElectionSpeechFallback,
)

# test_vote_flow.py
from tests.runtime.test_vote_flow import (
    TestVoteLifecycle,
    TestAntiStallPolicy,
)

# test_event_sourcing.py
from tests.runtime.test_event_sourcing import (
    TestPauseResumeEventSourcing,
    TestStartGameEventSourcing,
)

# test_judge_flow.py
from tests.runtime.test_judge_flow import (
    TestJudgeControlsNightRoleSequence,
    TestJudgeControlsDaySequence,
)

# test_strategy_directives.py
from tests.runtime.test_strategy_directives import (
    TestWitchStrategyHints,
    TestSeerStrategyDirectives,
    TestHunterStrategyDirectives,
    TestHybridStrategyDirectives,
    TestVillagerStrategyDirectives,
    TestIdiotStrategyDirectives,
    TestWolfStrategyDirectives,
    TestEmptySpeechGuard,
)