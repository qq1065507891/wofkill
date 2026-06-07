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



# ---------------------------------------------------------------------------
# Wolf consensus scripted tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Issue 2 (Task 5): Solo-wolf fallback target heuristic
# ---------------------------------------------------------------------------


class TestSoloWolfFallbackTarget:
    """Issue 2: When only 1 wolf is alive, plan should default to claimed Seer
    or day_push_target even without team discussion evidence."""

    def test_solo_wolf_default_targets_claimed_seer(self) -> None:
        """When only 1 wolf is alive and no plan, default kill target = claimed Seer."""
        from werewolf_agent.runtime.nodes._shared import _build_wolf_team_plan

        players = {
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id="p03", role="seer", alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
        }
        gs = GameState(
            game_id="solo_wolf_seer_test",
            players=players,
            night_number=3,
            day_number=2,
            events=[
                GameEvent(type="speech", payload={"speaker": "p03", "text": "我是预言家"}),
            ],
        )
        plan = _build_wolf_team_plan(gs, previous_plan=None)
        # Solo wolf (1 alive) should default to claimed Seer p03
        assert plan.get("night_kill_primary") == "p03", (
            f"expected default target p03 (claimed seer), got {plan.get('night_kill_primary')}"
        )

    def test_solo_wolf_no_claimed_seer_uses_day_push(self) -> None:
        """When no claimed Seer, default to day_push_target from previous plan."""
        from werewolf_agent.runtime.nodes._shared import _build_wolf_team_plan

        players = {
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
        }
        gs = GameState(
            game_id="solo_wolf_day_push_test",
            players=players,
            night_number=3,
            day_number=2,
            events=[],
        )
        prev_plan = {"day_push_target": "p05", "night_kill_primary": None}
        plan = _build_wolf_team_plan(gs, previous_plan=prev_plan)
        # No claimed Seer → use day_push_target
        assert plan.get("night_kill_primary") == "p05"


# ---------------------------------------------------------------------------
# P0-G3223805846-2: Wolf prompt must include live "已跳预言家" list sourced
# from day_speech events, not from stale model memory.  Without this, the
# wolf prompt hallucinates "p07 跳预言家" even though p07 was a villager
# who never publicly claimed seer (N2 hallucination).
# ---------------------------------------------------------------------------


class TestWolfDirectiveLiveSeerClaimants:
    """P0-G3223805846-2: 狼 prompt 中'已跳预言家'必须基于 day_speech 实时事件。"""

    def test_wolf_directive_contains_only_live_seer_claimants(self):
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.directives.wolf import build_wolf_directive

        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p01"] = PlayerState(id="p01", role="seer", alive=True)
        alive["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
        gs = GameState(
            players=alive, day_number=1, night_number=1,
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "p01", "text": "我是预言家，第 1 夜验 p02 是狼人。"
                }),
                GameEvent(type="speech", payload={
                    "speaker": "p02", "text": "我是预言家，第 1 夜验 p01 是好人（金水）。"
                }),
                GameEvent(type="speech", payload={
                    "speaker": "p07", "text": "我站边 p01，p02 是悍跳。"
                }),
            ],
        )
        d = build_wolf_directive(gs, "p01", wolf_team_plan=None)
        block = d.get("wolf_live_seer_claimants", "")
        # Positive marker: the live claimants block is present
        assert "已公开跳预言家" in block, f"missing live seer claimants block: {block!r}"
        # p01 and p02 must be listed (both publicly claimed)
        assert "p01" in block and "p02" in block, f"claimants missing: {block!r}"
        # p07 (villager, never claimed) must NOT be in the claimants list
        assert "p07" not in block, f"p07 wrongly listed as claimant: {block!r}"


# ---------------------------------------------------------------------------
# Issue 6 (Task B6): _planned_wolf_kill must validate primary alive;
# if primary is unreachable, force backup selection regardless of evidence.
# ---------------------------------------------------------------------------


class TestPlannedWolfKillPrimaryAlive:
    """P1-G3223805846-B6: wolf plan primary 必须存活，否则从 backup 选。

    背景：狼队讨论后形成 night_kill_primary + night_kill_backup 计划。
    但从狼队讨论到 wolf_consensus 执行击杀之间，primary 可能因白天投票
    出局而死亡。旧逻辑仅依赖 _first_alive_target 跳过死人 primary，然后
    用相同的 evidence 检查判断 backup；当 backup 没有证据且 quality 非
    strong 时，函数错误返回 None，调用方跌回 _legacy_wolf_consensus 随机
    选人。修复：当 primary 不可达时，强制使用 backup（绕过 evidence 校验）。
    """

    def _make_state(
        self,
        *,
        primary: str | None,
        backup: str | None,
        primary_alive: bool,
        backup_alive: bool,
        evidence_quality: str = "strong",
        evidence_targets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a minimal RuntimeState for _planned_wolf_kill.

        evidence_targets lists which targets appear in evidence_from_discussion.
        """
        players: dict[str, PlayerState] = {
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "p07": PlayerState(id="p07", role="villager", alive=primary_alive),
            "p03": PlayerState(id="p03", role="villager", alive=backup_alive),
            "p08": PlayerState(id="p08", role="villager", alive=True),
        }
        gs = GameState(
            game_id="planned_wolf_kill_primary_alive",
            players=players,
            night_number=1,
        )
        evidence = [{"target": t} for t in (evidence_targets or [])]
        return {
            "game_state": gs,
            "wolf_team_plan": {
                "night_kill_primary": primary,
                "night_kill_backup": backup,
                "evidence_quality": evidence_quality,
                "evidence_from_discussion": evidence,
            },
        }

    def test_skips_dead_primary_picks_alive_backup(self) -> None:
        """Primary 死亡时，强制使用 backup（即使 backup 无 evidence）。"""
        from werewolf_agent.runtime.nodes._shared import _planned_wolf_kill

        state = self._make_state(
            primary="p07",
            backup="p03",
            primary_alive=False,   # primary 死了
            backup_alive=True,     # backup 还活着
            evidence_quality="weak",
            evidence_targets=[],   # backup 也没有 evidence
        )
        result = _planned_wolf_kill(state)
        assert result is not None, "primary 死亡时应强制返回 backup"
        assert result["wolf_kill_target_id"] == "p03", (
            f"expected backup p03 (primary p07 dead), got {result['wolf_kill_target_id']}"
        )

    def test_returns_alive_primary_when_present(self) -> None:
        """Primary 存活时优先使用 primary。"""
        from werewolf_agent.runtime.nodes._shared import _planned_wolf_kill

        state = self._make_state(
            primary="p07",
            backup="p03",
            primary_alive=True,    # primary 活着
            backup_alive=True,
            evidence_quality="strong",
            evidence_targets=["p07"],
        )
        result = _planned_wolf_kill(state)
        assert result is not None
        assert result["wolf_kill_target_id"] == "p07", (
            f"expected primary p07, got {result['wolf_kill_target_id']}"
        )

    def test_returns_none_when_both_dead(self) -> None:
        """Primary 和 backup 都死亡时返回 None（让调用方走 solo-wolf fallback）。"""
        from werewolf_agent.runtime.nodes._shared import _planned_wolf_kill

        state = self._make_state(
            primary="p07",
            backup="p03",
            primary_alive=False,   # 都死了
            backup_alive=False,
            evidence_quality="strong",
            evidence_targets=["p07", "p03"],
        )
        result = _planned_wolf_kill(state)
        assert result is None, f"expected None when both dead, got {result}"

    def test_dead_primary_with_evidence_only_for_primary_still_picks_backup(self) -> None:
        """Primary 死亡、evidence 只列了 primary 时，函数不应死锁，应选 backup。"""
        from werewolf_agent.runtime.nodes._shared import _planned_wolf_kill

        state = self._make_state(
            primary="p07",
            backup="p03",
            primary_alive=False,
            backup_alive=True,
            evidence_quality="weak",
            evidence_targets=["p07"],  # 只有死人 primary 在 evidence 里
        )
        result = _planned_wolf_kill(state)
        assert result is not None, "primary 死亡时即使 evidence 不含 backup 也应返回 backup"
        assert result["wolf_kill_target_id"] == "p03"
