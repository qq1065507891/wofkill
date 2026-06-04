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
# Vote resolution standalone tests
# ---------------------------------------------------------------------------

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
                suspect_reason="p02发言前后矛盾",
                not_voting_reason="其他人没明显证据",
                private_reason="我投p02",
            ), RetryInfo()

    agent = Agent()

    class Registry:
        def get_agent(self, player_id):
            return agent

    result = agent_day_vote({"game_state": gs}, _new_engine(), Registry(), "p01")

    assert result["vote_target"] == "p02"
    assert agent.context.legal_targets == ["p02", "p03"]

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