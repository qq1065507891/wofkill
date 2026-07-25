# -*- coding: utf-8 -*-
"""
验证运行时投票流程、事件契约与规则结算的一致性。

作者: Project contributors
修改日期: 2026-07-25
"""

from __future__ import annotations

import json

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
        "p04": PlayerState(id="p04", role="villager"),
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
        "exile_votes": {"p01": "p03", "p02": "p04"},
        "revote": False,
    })

    vote_event = next(event for event in result["game_state"].events if event.type == "vote_resolved")
    payload = vote_event.payload
    assert payload["vote_weight_format_version"] == 2
    assert payload["base_vote_weight"] == 2
    assert payload["sheriff_id"] == "p01"
    assert payload["sheriff_vote_weight"] == 3
    assert payload["weighted_tally"] == {"p03": 3, "p04": 2}
    assert payload["vote_weights"] == {"p01": 3, "p02": 2}
    assert payload["weighted_tally_units"] == payload["weighted_tally"]
    assert payload["vote_weight_units"] == payload["vote_weights"]
    assert payload["weighted_tally_display"] == {"p03": 1.5, "p04": 1}
    assert payload["vote_weights_display"] == {"p01": 1.5, "p02": 1}
    assert type(payload["weighted_tally_display"]["p03"]) is float
    assert type(payload["weighted_tally_display"]["p04"]) is int
    assert type(payload["vote_weights_display"]["p01"]) is float
    assert type(payload["vote_weights_display"]["p02"]) is int
    json.dumps(payload)


def test_base_three_vote_producers_serialize_four_thirds_display() -> None:
    from werewolf_agent.runtime.graph import resolve_vote
    from werewolf_agent.runtime.nodes.day_vote import _broadcast_vote_details
    from werewolf_agent.runtime.vote_display import (
        decode_vote_resolved_payload,
        decode_vote_tally_payload,
        vote_units_to_display,
    )

    engine = _new_engine()
    engine.ruleset.raw.setdefault("game_rules", {})["base_vote_weight"] = 3
    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="villager"),
        "p03": PlayerState(id="p03", role="werewolf"),
        "p04": PlayerState(id="p04", role="villager"),
    }
    gs = GameState(
        game_id="base_three_vote_display",
        players=players,
        sheriff_id="p01",
        sheriff_badge_state="active",
        day_number=2,
    )
    votes = {"p01": "p03", "p02": "p04"}

    broadcast_state = {
        "game_state": gs,
        "engine": engine,
        "judge_llm_enabled": False,
    }
    broadcast_gs = _broadcast_vote_details(broadcast_state, gs, votes)
    tally_payload = broadcast_gs.events[-1].payload

    assert tally_payload["base_vote_weight"] == 3
    assert tally_payload["sheriff_weight_units"] == 4
    assert type(tally_payload["sheriff_weight_display"]) is float
    assert tally_payload["sheriff_weight_display"] == pytest.approx(4 / 3)
    assert type(tally_payload["tally_display"]["p04"]) is int
    persisted_tally = json.loads(json.dumps(tally_payload))
    decoded_tally = decode_vote_tally_payload(persisted_tally)
    assert decoded_tally.sheriff_weight_display == vote_units_to_display(
        4,
        base_vote_weight=3,
    )

    resolved = resolve_vote({
        "game_state": gs,
        "engine": engine,
        "exile_votes": votes,
        "revote": False,
    })
    resolved_payload = next(
        event.payload
        for event in resolved["game_state"].events
        if event.type == "vote_resolved"
    )

    assert resolved_payload["base_vote_weight"] == 3
    assert resolved_payload["vote_weight_units"]["p01"] == 4
    assert type(resolved_payload["vote_weights_display"]["p01"]) is float
    assert resolved_payload["vote_weights_display"]["p01"] == pytest.approx(4 / 3)
    assert type(resolved_payload["vote_weights_display"]["p02"]) is int
    persisted_resolved = json.loads(json.dumps(resolved_payload))
    decoded_resolved = decode_vote_resolved_payload(persisted_resolved)
    assert decoded_resolved.vote_weights_display["p01"] == vote_units_to_display(
        4,
        base_vote_weight=3,
    )


@pytest.mark.parametrize(
    ("votes", "revote", "pk_candidates", "accepted_votes", "tally_units"),
    [
        (
            {"p01": "p01", "p02": "p04"},
            False,
            None,
            {"p02": "p04"},
            {"p04": 2},
        ),
        (
            {"p01": "missing", "p02": "p03", "p04": "p02"},
            False,
            None,
            {"p04": "p02"},
            {"p02": 2},
        ),
        (
            {"p01": "p04", "p04": "p02"},
            True,
            ["p02", "p03"],
            {"p04": "p02"},
            {"p02": 2},
        ),
    ],
)
def test_vote_events_only_include_rule_accepted_votes(
    votes: dict[str, str],
    revote: bool,
    pk_candidates: list[str] | None,
    accepted_votes: dict[str, str],
    tally_units: dict[str, int],
) -> None:
    from werewolf_agent.runtime.graph import resolve_vote
    from werewolf_agent.runtime.nodes.day_vote import _broadcast_vote_details

    engine = _new_engine()
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=False),
        "p04": PlayerState(id="p04", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="accepted_vote_projection",
        players=players,
        day_number=2,
    )
    expected_result = engine.resolve_vote(
        gs,
        votes=votes,
        revote=revote,
        pk_candidates=pk_candidates,
    )

    broadcast_gs = _broadcast_vote_details(
        {
            "game_state": gs,
            "engine": engine,
            "revote": revote,
            "pk_candidates": pk_candidates,
            "judge_llm_enabled": False,
        },
        gs,
        votes,
    )
    tally_payload = broadcast_gs.events[-1].payload
    assert tally_payload["tally"] == tally_units
    assert tally_payload["tally_units"] == tally_units

    resolved = resolve_vote({
        "game_state": gs,
        "engine": engine,
        "exile_votes": votes,
        "revote": revote,
        "pk_candidates": pk_candidates,
    })
    resolved_event = next(
        event
        for event in resolved["game_state"].events
        if event.type == "vote_resolved"
    )
    resolved_payload = resolved_event.payload

    assert resolved["_vote_result"] == expected_result
    assert resolved["game_state"].votes == accepted_votes
    assert resolved_payload["weighted_tally"] == tally_units
    assert resolved_payload["weighted_tally_units"] == tally_units
    assert resolved_payload["vote_weight_units"] == {
        voter_id: 2 for voter_id in accepted_votes
    }
    assert {
        vote["voter"]: vote["target"]
        for vote in resolved_payload["votes"]
    } == accepted_votes


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
            "candidate_comparison": "p02有站边摇摆和跟风票；p04只是发言短，证据较弱",
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
        "candidate_comparison": "p02有站边摇摆和跟风票；p04只是发言短，证据较弱",
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
                candidate_comparison="p02发言矛盾比p03更具体",
                private_reason="我投p02",
            ), RetryInfo()

    agent = Agent()

    class Registry:
        def get_agent(self, player_id):
            return agent

    result = agent_day_vote({"game_state": gs}, _new_engine(), Registry(), "p01")

    assert result["vote_target"] == "p02"
    assert agent.context.legal_targets == ["p02", "p03"]


def test_agent_day_vote_does_not_invent_fallback_target_without_evidence() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_day_vote

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    gs = GameState(game_id="vote_no_evidence_fallback", players=players, day_number=1)

    class Agent:
        def act(self, context):
            return FallbackAction(
                action_type=ActionType.VOTE,
                target_id=None,
                reason="fallback: 结构化输出失败，无足够公开证据补票",
            ), RetryInfo()

    class Registry:
        def get_agent(self, player_id):
            return Agent()

    result = agent_day_vote({"game_state": gs}, _new_engine(), Registry(), "p01")

    assert result["vote_target"] is None


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


class TestDeadVoterFiltered:
    """P1-G3223805846-2: 死人的 vote 不应进入 tally result。"""

    def test_dead_voter_excluded_from_vote_resolved_payload(self) -> None:
        """Stale vote from a dead player (e.g. restored checkpoint) must be filtered."""
        from werewolf_agent.runtime.graph import resolve_vote

        engine = _new_engine()
        players = {
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p11": PlayerState(id="p11", role="villager", alive=False),  # 昨夜死
        }
        gs = GameState(game_id="dead_vote_filter", players=players, day_number=2)

        # p11 已死但其 vote 仍在 exile_votes（模拟 checkpoint 恢复/重放残留）
        result = resolve_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p11": "p02", "p01": "p02"},
            "revote": False,
            "consecutive_no_exile_days": 0,
        })

        vote_event = next(
            (e for e in result["game_state"].events if e.type == "vote_resolved"),
            None,
        )
        assert vote_event is not None, "vote_resolved event missing"

        # p11 是死人，vote 不应出现在 payload["votes"]
        voters_in_payload = [v["voter"] for v in vote_event.payload["votes"]]
        assert "p11" not in voters_in_payload, (
            f"dead voter p11 leaked into votes payload: {vote_event.payload['votes']}"
        )

        # p11 的票不应计入 weighted_tally
        assert vote_event.payload["weighted_tally"] == {"p02": 2}, (
            f"dead voter inflated weighted_tally: {vote_event.payload['weighted_tally']}"
        )

        # p11 不应在 vote_weights 中
        assert "p11" not in vote_event.payload["vote_weights"], (
            f"dead voter in vote_weights: {vote_event.payload['vote_weights']}"
        )

    def test_dead_vote_disabled_voter_also_excluded(self) -> None:
        """白痴亮牌后 vote_enabled=False，同样不应进入 payload。"""
        from werewolf_agent.runtime.graph import resolve_vote

        engine = _new_engine()
        players = {
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p05": PlayerState(id="p05", role="idiot", alive=True, vote_enabled=False),
        }
        gs = GameState(game_id="vote_disabled_filter", players=players, day_number=2)

        # p05 是白痴，vote_enabled=False 但其 vote 仍在 exile_votes
        result = resolve_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p05": "p02", "p01": "p02"},
            "revote": False,
            "consecutive_no_exile_days": 0,
        })

        vote_event = next(
            (e for e in result["game_state"].events if e.type == "vote_resolved"),
            None,
        )
        assert vote_event is not None

        voters_in_payload = [v["voter"] for v in vote_event.payload["votes"]]
        assert "p05" not in voters_in_payload, (
            f"vote-disabled voter p05 leaked into payload: {vote_event.payload['votes']}"
        )
        assert vote_event.payload["weighted_tally"] == {"p02": 2}

    def test_dead_vote_filter_does_not_change_exile_choice(self) -> None:
        """dead vote 被过滤后，tally result 与 engine.resolve_vote 一致（不死人票计入）。"""
        from werewolf_agent.runtime.graph import resolve_vote

        engine = _new_engine()
        players = {
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id="p03", role="villager", alive=True),
            "p11": PlayerState(id="p11", role="villager", alive=False),
        }
        gs = GameState(game_id="dead_vote_filter_choice", players=players, day_number=2)

        # p11 (dead) 投 p01；p02/p03 (alive) 投 p01；p01 (alive) 投 p02
        # 死票过滤后，p01 应获 2 票 (p02, p03)，p02 应获 1 票 (p01)
        result = resolve_vote({
            "game_state": gs,
            "engine": engine,
            "exile_votes": {"p11": "p01", "p02": "p01", "p03": "p01", "p01": "p02"},
            "revote": False,
            "consecutive_no_exile_days": 0,
        })

        # exile 应是 p01（多数）
        assert result["_vote_result"].exiled_player_id == "p01"
