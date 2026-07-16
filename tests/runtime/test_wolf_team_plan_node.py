"""Tests for wolf_team_plan_node (LLM-first plan generation with fallback)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from tests.runtime.test_agent_wolf_team_plan import _FakeAgent, _FakeModelRouter


def _make_gs(*, night=1, wolves=("p04", "p05", "p08", "p10"), with_discussion=True):
    players: dict[str, PlayerState] = {}
    for i in range(1, 13):
        pid = f"p{i:02d}"
        role = "werewolf" if pid in wolves else (
            "seer" if pid == "p06" else "villager"
        )
        players[pid] = PlayerState(id=pid, role=role, alive=True)
    events: list[GameEvent] = [GameEvent(type="enter_night", payload={"night": night})]
    if with_discussion:
        for w in wolves:
            events.append(GameEvent(
                type="wolf_discussion",
                payload={
                    "wolf_id": w,
                    "round": 1,
                    "night_number": night,
                    "text": f"{w} 同意刀 p01",
                    "visibility": "werewolf_team_only",
                },
            ))
    return GameState(
        game_id="test_wtpn", phase="night", night_number=night,
        players=players, events=events,
    )


class _MockRegistry:
    def __init__(self, agents=None):
        self._agents = agents or {}

    def get_agent(self, pid):
        return self._agents.get(pid)


def _valid_plan_dict(**overrides):
    base = dict(
        night_number=1,
        night_kill_primary="p01",
        night_kill_backup=None,
        fake_seer="p04",
        pusher="p05",
        hooker="p08",
        deep_cover="p10",
        public_story="跟刀口推 p01",
        evidence_quality="strong",
        reasoning="夜聊一致",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------

class TestNoRegistryFallback:
    def test_no_registry_uses_fallback(self):
        from werewolf_agent.runtime.nodes.night import wolf_team_plan_node
        gs = _make_gs()
        state = {"game_state": gs}
        result = wolf_team_plan_node(state)
        plan = result["wolf_team_plan"]
        assert plan["consensus_method"] == "fallback"
        # fallback emits wolf_team_plan_fallback event
        fallback_events = [
            e for e in result["game_state"].events
            if e.type == "wolf_team_plan_fallback"
        ]
        assert len(fallback_events) == 1
        assert fallback_events[0].payload["reason"] == "no_registry"

    def test_no_registry_does_not_reuse_stale_llm_failure_metadata(self):
        from werewolf_agent.runtime.nodes.night import wolf_team_plan_node
        gs = _make_gs()
        state = {
            "game_state": gs,
            "wolf_team_plan_failure": {
                "reason": "empty_response",
                "stage": "model_output",
                "attempts": 3,
                "last_error": "empty_response",
                "captain_id": "p04",
            },
        }

        result = wolf_team_plan_node(state)

        fallback_event = next(
            e for e in result["game_state"].events
            if e.type == "wolf_team_plan_fallback"
        )
        assert fallback_event.payload == {
            "night_number": 1,
            "reason": "no_registry",
        }
        assert fallback_event.visibility is EventVisibility.WEREWOLF_TEAM_ONLY

    def test_no_alive_wolves_returns_empty(self):
        from werewolf_agent.runtime.nodes.night import wolf_team_plan_node
        gs = _make_gs()
        gs = replace(gs, players={
            pid: replace(p, alive=False) if p.role == "werewolf" else p
            for pid, p in gs.players.items()
        })
        result = wolf_team_plan_node({"game_state": gs})
        assert "wolf_team_plan" not in result
        # No new events
        new_events = [e for e in result["game_state"].events if e.type.startswith("wolf_team_plan")]
        assert len(new_events) == 0


class TestLLMSuccess:
    def test_llm_success_uses_llm_plan(self):
        from werewolf_agent.runtime.nodes import night as night_mod
        gs = _make_gs()
        registry = _MockRegistry()  # not actually used because we mock the agent function
        state = {"game_state": gs, "agent_registry": registry}

        llm_plan = _valid_plan_dict()
        llm_plan["consensus_method"] = "llm"
        llm_plan["captain_id"] = "p04"

        with patch.object(night_mod, "_alive_wolves") as mock_alive, \
             patch("werewolf_agent.runtime.agent_adapter.agent_wolf_team_plan",
                   return_value=llm_plan):
            mock_alive.return_value = ["p04", "p05", "p08", "p10"]
            result = night_mod.wolf_team_plan_node(state)

        plan = result["wolf_team_plan"]
        assert plan["consensus_method"] == "llm"
        assert plan["captain_id"] == "p04"

        # Only wolf_team_plan event, no fallback event
        plan_events = [e for e in result["game_state"].events if e.type == "wolf_team_plan"]
        fallback_events = [e for e in result["game_state"].events if e.type == "wolf_team_plan_fallback"]
        assert len(plan_events) == 1
        assert len(fallback_events) == 0

    def test_llm_plan_event_has_visibility(self):
        from werewolf_agent.runtime.nodes import night as night_mod
        gs = _make_gs()
        state = {"game_state": gs, "agent_registry": _MockRegistry()}

        llm_plan = _valid_plan_dict()
        llm_plan["consensus_method"] = "llm"
        with patch("werewolf_agent.runtime.agent_adapter.agent_wolf_team_plan", return_value=llm_plan):
            result = night_mod.wolf_team_plan_node(state)
        evt = next(e for e in result["game_state"].events if e.type == "wolf_team_plan")
        assert evt.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
        assert evt.schema_version == "2"


class TestLLMFallback:
    def test_llm_returns_none_triggers_fallback(self):
        from werewolf_agent.runtime.nodes import night as night_mod
        gs = _make_gs()
        state = {"game_state": gs, "agent_registry": _MockRegistry()}

        with patch("werewolf_agent.runtime.agent_adapter.agent_wolf_team_plan", return_value=None):
            result = night_mod.wolf_team_plan_node(state)

        plan = result["wolf_team_plan"]
        assert plan["consensus_method"] == "fallback"

        fallback_events = [e for e in result["game_state"].events if e.type == "wolf_team_plan_fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0].payload["reason"] == "llm_failed_or_unavailable"
        assert fallback_events[0].schema_version == "2"
        assert fallback_events[0].visibility is EventVisibility.WEREWOLF_TEAM_ONLY

    def test_llm_raises_exception_triggers_fallback(self):
        from werewolf_agent.runtime.nodes import night as night_mod
        gs = _make_gs()
        state = {"game_state": gs, "agent_registry": _MockRegistry()}

        with patch(
            "werewolf_agent.runtime.agent_adapter.agent_wolf_team_plan",
            side_effect=RuntimeError("simulated provider crash"),
        ):
            result = night_mod.wolf_team_plan_node(state)

        plan = result["wolf_team_plan"]
        assert plan["consensus_method"] == "fallback"
        fallback_events = [e for e in result["game_state"].events if e.type == "wolf_team_plan_fallback"]
        assert len(fallback_events) == 1
        assert "agent_exception" in fallback_events[0].payload["reason"]

    def test_fallback_plan_has_standard_fields(self):
        """Fallback path must still produce a plan dict with the 9 canonical keys."""
        from werewolf_agent.runtime.nodes.night import wolf_team_plan_node
        gs = _make_gs()
        result = wolf_team_plan_node({"game_state": gs})
        plan = result["wolf_team_plan"]
        for key in (
            "fake_seer", "pusher", "hooker", "deep_cover",
            "night_kill_primary", "night_kill_backup",
            "public_story", "evidence_quality",
        ):
            assert key in plan

    def test_fallback_event_uses_llm_failure_metadata(self):
        from werewolf_agent.runtime.nodes import night as night_mod
        gs = _make_gs()
        router = _FakeModelRouter([("", None), ("", None), ("", None)])
        registry = _MockRegistry({"p04": _FakeAgent("p04", router)})
        state = {"game_state": gs, "agent_registry": registry}

        with patch.object(night_mod, "_alive_wolves") as mock_alive:
            mock_alive.return_value = ["p04", "p05", "p08", "p10"]
            result = night_mod.wolf_team_plan_node(state)

        fallback_event = next(
            e for e in result["game_state"].events
            if e.type == "wolf_team_plan_fallback"
        )
        assert fallback_event.payload["reason"] == "empty_response"
        assert fallback_event.payload["stage"] == "model_output"
        assert fallback_event.payload["attempts"] == 3
        assert fallback_event.payload["last_error"] == "empty_response"
        assert fallback_event.payload["captain_id"] == "p04"


class TestStateReturn:
    def test_returns_game_state_with_events_appended(self):
        from werewolf_agent.runtime.nodes.night import wolf_team_plan_node
        gs = _make_gs()
        original_event_count = len(gs.events)
        result = wolf_team_plan_node({"game_state": gs})
        new_event_count = len(result["game_state"].events)
        # At least wolf_team_plan event appended (fallback path also adds fallback event)
        assert new_event_count > original_event_count

    def test_plan_dict_in_return(self):
        from werewolf_agent.runtime.nodes.night import wolf_team_plan_node
        gs = _make_gs()
        result = wolf_team_plan_node({"game_state": gs})
        assert "wolf_team_plan" in result
        assert isinstance(result["wolf_team_plan"], dict)
