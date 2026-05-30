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
