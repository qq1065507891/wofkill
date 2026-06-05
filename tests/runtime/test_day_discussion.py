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



# ---------------------------------------------------------------------------
# Free discussion (day speech)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Day speech agent interactions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Day flow announcements and routing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Night death last words
# ---------------------------------------------------------------------------

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


def test_announce_deaths_resets_speech_order() -> None:
    """D2+ must NOT reuse D1's speech_order; announce_deaths resets it."""
    from werewolf_agent.runtime.graph import announce_deaths

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    # D1 with speech_order pre-populated from previous flow
    gs_d1 = GameState(
        game_id="d1_announce",
        players=players,
        phase="day",
        day_number=1,
        night_number=1,
    )
    d1_result = announce_deaths({
        "game_state": gs_d1,
        "speech_order": ["p01", "p02", "p03"],
    })
    # After D1 announce, speech_order should be empty (reset for fresh generation)
    assert d1_result["speech_order"] == [], (
        f"D1 announce_deaths should reset speech_order, got {d1_result.get('speech_order')}"
    )
    assert d1_result["speech_index"] == 0
    assert d1_result["current_speaker_id"] is None

    # D2 with stale D1 speech_order
    gs_d2 = GameState(
        game_id="d2_announce",
        players=players,
        phase="day",
        day_number=2,
        night_number=2,
    )
    d2_result = announce_deaths({
        "game_state": gs_d2,
        "speech_order": ["p01", "p02", "p03"],  # Stale from D1
    })
    assert d2_result["speech_order"] == [], (
        "D2 announce_deaths should reset stale speech_order, "
        f"got {d2_result.get('speech_order')}"
    )