"""End-to-end smoke: real wolf_discussion → wolf_team_plan_node → wolf_consensus
chain with mock LLM that returns a valid WolfTeamPlan JSON.

Verifies the new LLM-path produces a wolf_team_plan event with
consensus_method='llm' and that wolf_consensus picks the captain's chosen
primary kill target.

This substitutes for the real-LLM game smoke test (T9 of plan) when the
remote provider is rate-limited (HTTP 429 baidu/qianfan).
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.nodes import night as night_mod


def _new_engine() -> RuleEngine:
    return RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")


class _ScriptedRouter:
    """Mock model_router that returns a pre-defined WolfTeamPlan JSON."""

    def __init__(self, plan_payload: dict[str, Any]) -> None:
        self.plan_payload = plan_payload
        self.call_count = 0

    def generate(self, *, agent_id, task_type, prompt, system_prompt, tools, tool_choice):
        self.call_count += 1

        class _Result:
            def __init__(self, text):
                self.text = text
                self.tool_call_received = True
        return _Result(json.dumps(self.plan_payload, ensure_ascii=False))


class _ScriptedAgent:
    def __init__(self, agent_id, model_router):
        self.agent_id = agent_id
        self.model_router = model_router


class _DispatchStubbedRegistry:
    """Returns a scripted agent only for the captain (sorted(alive_wolves)[0])."""

    def __init__(self, captain_id, router):
        self._captain_id = captain_id
        self._router = router

    def get_agent(self, pid):
        if pid == self._captain_id:
            return _ScriptedAgent(pid, self._router)
        return None


def test_e2e_n1_llm_captain_produces_plan_and_consensus_picks_target(monkeypatch):
    """Full N1 chain: wolf_discussion → wolf_team_plan_node → wolf_consensus.

    The captain LLM returns a valid WolfTeamPlan that names p06 (alive seer)
    as primary kill. wolf_consensus must pick p06 via _planned_wolf_kill.
    """
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    # Find alive wolves to compute captain
    alive_wolves = sorted(pid for pid, p in players.items() if p.role == "werewolf" and p.alive)
    alive_non_wolves = sorted(pid for pid, p in players.items() if p.role != "werewolf" and p.alive)
    captain = alive_wolves[0]
    target = alive_non_wolves[0]
    backup_target = alive_non_wolves[1]

    plan_payload = {
        "night_number": 1,
        "night_kill_primary": target,
        "night_kill_backup": backup_target,
        "fake_seer": alive_wolves[0],
        "pusher": alive_wolves[1],
        "hooker": alive_wolves[2],
        "deep_cover": alive_wolves[3],
        "public_story": "跟刀口推第一刀目标",
        "evidence_quality": "strong",
        "reasoning": "队友一致同意刀首位非狼,角色按 alive 顺序分配",
    }
    router = _ScriptedRouter(plan_payload)
    registry = _DispatchStubbedRegistry(captain, router)

    # Stub wolf_discussion's dispatch so it doesn't try to call real LLM
    def fake_dispatch_agent(_state, _fn, *_extra_args, **_kwargs):
        wolf_id = _extra_args[0]
        return {"speech_text": f"{wolf_id} round {_state.get('wolf_discussion_round')}: 刀 {target}, 分工见队长"}

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch_agent)

    gs = GameState(game_id="e2e_n1", players=players, night_number=1, phase="night")
    state = {
        "game_state": gs,
        "engine": engine,
        "agent_registry": registry,
    }

    # Step 1: wolf_discussion produces 12 discussion events (4 wolves × 3 rounds)
    state = night_mod.wolf_discussion(state)
    state["game_state"]  # noqa — sanity check key exists
    assert "wolf_team_plan" not in state, "wolf_discussion must not emit plan"
    disc_events = [e for e in state["game_state"].events if e.type == "wolf_discussion"]
    assert len(disc_events) >= 4  # at least one full round; mid-loop may end early

    # Step 2: wolf_team_plan_node calls LLM captain, gets structured plan
    state2 = {**state, "agent_registry": registry, "engine": engine}
    state2 = night_mod.wolf_team_plan_node(state2)

    assert router.call_count == 1, "LLM router should be called exactly once"
    plan = state2["wolf_team_plan"]
    assert plan["consensus_method"] == "llm"
    assert plan["captain_id"] == captain
    assert plan["fake_seer"] == alive_wolves[0]
    assert plan["pusher"] == alive_wolves[1]
    assert plan["night_kill_primary"] == target

    plan_events = [e for e in state2["game_state"].events if e.type == "wolf_team_plan"]
    fallback_events = [e for e in state2["game_state"].events if e.type == "wolf_team_plan_fallback"]
    assert len(plan_events) == 1
    assert plan_events[0].payload["visibility"] == "werewolf_team_only"
    assert plan_events[0].payload["consensus_method"] == "llm"
    assert len(fallback_events) == 0

    # Step 3: wolf_consensus reads the plan and picks the captain's target
    from werewolf_agent.runtime.graph import wolf_consensus
    state3 = {**state2, "engine": engine}
    result = wolf_consensus(state3)
    assert result["wolf_kill_target_id"] == target, (
        f"wolf_consensus should pick captain's planned target {target}, "
        f"got {result.get('wolf_kill_target_id')}"
    )

    # Verify wolf_kill_selected event
    kill_events = [e for e in result["game_state"].events if e.type == "wolf_kill_selected"]
    assert len(kill_events) == 1
    assert kill_events[0].payload["target_id"] == target


def test_e2e_n1_llm_failure_triggers_fallback_with_audit_event(monkeypatch):
    """If LLM captain agent is missing, fallback fires, audit event emitted."""
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=2)

    def fake_dispatch_agent(_state, _fn, *_extra_args, **_kwargs):
        return {"speech_text": "discussion text"}

    monkeypatch.setattr(night_mod, "_dispatch_agent", fake_dispatch_agent)

    gs = GameState(game_id="e2e_n1_fail", players=players, night_number=1, phase="night")

    class EmptyRegistry:
        def get_agent(self, pid):
            return None

    state = {
        "game_state": gs,
        "engine": engine,
        "agent_registry": EmptyRegistry(),
    }
    state = night_mod.wolf_discussion(state)
    state = night_mod.wolf_team_plan_node(state)

    plan = state["wolf_team_plan"]
    assert plan["consensus_method"] == "fallback"

    fallback_events = [e for e in state["game_state"].events if e.type == "wolf_team_plan_fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0].payload["reason"] in ("no_registry", "llm_failed_or_unavailable")
