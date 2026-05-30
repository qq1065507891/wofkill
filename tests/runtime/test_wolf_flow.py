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



def _last_non_broadcast_event(gs: GameState) -> GameEvent:
    return next(e for e in reversed(gs.events) if e.type != "judge_broadcast")


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
        event = _last_non_broadcast_event(result["game_state"])
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
# F1: Wolf role assignment deduplication
# ---------------------------------------------------------------------------


class TestWolfPlanDedup:
    """F1: Wolf role assignment deduplication."""

    def test_consensus_no_duplicate_roles(self) -> None:
        """Same wolf proposed for two roles should only get the first."""
        from werewolf_agent.runtime.wolf_strategy import summarize_wolf_consensus
        events = [
            GameEvent(type="wolf_discussion", payload={
                "wolf_id": "p01", "round": 1, "night_number": 1,
                "text": "我做假预言家，p01也做倒钩",
            }),
            GameEvent(type="wolf_discussion", payload={
                "wolf_id": "p02", "round": 2, "night_number": 1,
                "text": "同意p01做假预言家",
            }),
        ]
        result = summarize_wolf_consensus(events, ["p01", "p02"], night_number=1)
        # p01 should be fake_seer only, not also hooker
        assert result.get("fake_seer") == "p01"
        assert result.get("hooker") != "p01"

    def test_plan_dedup_on_merge(self) -> None:
        """build_wolf_team_plan_from_discussion should not assign same wolf twice."""
        from werewolf_agent.runtime.wolf_strategy import build_wolf_team_plan_from_discussion
        gs = GameState(game_id="dedup_test", phase="night", night_number=1)
        consensus = {"fake_seer": "p01", "hooker": "p01", "evidence_quality": "strong"}
        plan = build_wolf_team_plan_from_discussion(gs, previous_plan=None, consensus=consensus)
        assigned = [plan.get(r) for r in ("fake_seer", "pusher", "hooker", "deep_cover") if plan.get(r)]
        assert len(assigned) == len(set(assigned)), f"Duplicate assignment: {assigned}"



# ---------------------------------------------------------------------------
# F3: Wolf fallback vote should not target teammates
# ---------------------------------------------------------------------------


class TestWolfFallbackVoteNoTeammate:
    """F3: Wolf fallback vote should not target teammates."""

    def test_choose_vote_fallback_excludes_wolf_teammates(self) -> None:
        from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
        players = {
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "w2": PlayerState(id="w2", role="werewolf", alive=True),
            "v1": PlayerState(id="v1", role="villager", alive=True),
            "v2": PlayerState(id="v2", role="villager", alive=True),
        }
        gs = GameState(game_id="wolf_vote_test", players=players, phase="day", day_number=1)
        result = choose_vote_fallback_target(gs, "w1", ["w2", "v1", "v2"])
        assert result != "w2", "Wolf fallback vote should not target teammate"
        assert result in ("v1", "v2")