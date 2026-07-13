"""Tests for agent_wolf_team_plan (LLM captain decision adapter).

Covers:
- captain selection (sorted(alive_wolves)[0])
- captain agent missing → returns None (caller fallback)
- empty alive wolves → None
- LLM returns valid JSON → parsed plan with consensus_method='llm'
- LLM returns invalid JSON → retry, eventually None
- LLM returns schema-invalid JSON → retry
- LLM target points to dead/non-wolf → membership rejected, retry
- LLM NotImplementedError → immediate None (provider unsupported)
- discussion text correctly collected from gs.events for current night
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.agent_adapter import (
    _extract_first_balanced_json_object,
    agent_wolf_team_plan,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_gs(*, night=1, wolves=("p04", "p05", "p08", "p10"), extra_events=()):
    players: dict[str, PlayerState] = {}
    for i in range(1, 13):
        pid = f"p{i:02d}"
        role = "werewolf" if pid in wolves else (
            "seer" if pid == "p06" else "villager"
        )
        players[pid] = PlayerState(id=pid, role=role, alive=True)
    events: list[GameEvent] = [GameEvent(type="enter_night", payload={"night": night})]
    # Add 4 wolf_discussion entries for current night
    for round_num in (1,):
        for w in wolves:
            events.append(GameEvent(
                type="wolf_discussion",
                payload={
                    "wolf_id": w,
                    "round": round_num,
                    "night_number": night,
                    "text": f"我是 {w}, 建议刀 p01",
                    "visibility": "werewolf_team_only",
                },
            ))
    events.extend(extra_events)
    return GameState(
        game_id="test_wtp", phase="night", night_number=night,
        players=players, events=events,
    )


class _FakeModelRouter:
    """Mock model router; .generate() returns a pre-configured GenerateResult-shaped object."""

    def __init__(self, responses):
        # responses: list of (text, exception_or_none) — consumed in order
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        agent_id,
        task_type,
        prompt,
        system_prompt,
        tools=None,
        tool_choice=None,
    ):
        self.calls.append({
            "agent_id": agent_id, "task_type": task_type, "prompt": prompt,
            "system_prompt": system_prompt, "tool_choice": tool_choice,
        })
        if not self._responses:
            raise RuntimeError("test bug: no more mock responses")
        text, exc = self._responses.pop(0)
        if exc is not None:
            raise exc

        class _Result:
            def __init__(self, t):
                self.text = t
                self.tool_call_received = True
        return _Result(text)


class _FakeAgent:
    def __init__(self, agent_id, model_router):
        self.agent_id = agent_id
        self.model_router = model_router


class _FakeRegistry:
    def __init__(self, agents_by_id):
        self._agents = agents_by_id

    def get_agent(self, pid):
        return self._agents.get(pid)


def _valid_plan_json(**overrides):
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
        reasoning="夜聊一致同意刀 p01, 角色按 alive 顺序分配",
    )
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Captain selection
# ---------------------------------------------------------------------------

class TestCaptainSelection:
    def test_captain_is_lowest_alive_wolf_id(self):
        gs = _make_gs(wolves=("p04", "p05", "p08", "p10"))
        router = _FakeModelRouter([(_valid_plan_json(), None)])
        agents = {
            "p04": _FakeAgent("p04", router),
            "p05": _FakeAgent("p05", _FakeModelRouter([])),
        }
        registry = _FakeRegistry(agents)
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None
        assert plan["captain_id"] == "p04"
        # p04's router was called, p05's wasn't
        assert len(router.calls) == 1

    def test_captain_skipped_when_dead_uses_next_alive(self):
        gs = _make_gs(wolves=("p04", "p05", "p08", "p10"))
        # Kill p04
        gs = replace(gs, players={
            **gs.players,
            "p04": replace(gs.players["p04"], alive=False),
        })
        router = _FakeModelRouter([(_valid_plan_json(fake_seer="p05", pusher="p08", hooker="p10", deep_cover=None), None)])
        registry = _FakeRegistry({"p05": _FakeAgent("p05", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None
        assert plan["captain_id"] == "p05"

    def test_no_alive_wolves_returns_none(self):
        gs = _make_gs()
        gs = replace(gs, players={
            pid: replace(p, alive=False) if p.role == "werewolf" else p
            for pid, p in gs.players.items()
        })
        registry = _FakeRegistry({})
        assert agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry) is None

    def test_captain_agent_unavailable_returns_none(self):
        gs = _make_gs(wolves=("p04", "p05", "p08", "p10"))
        registry = _FakeRegistry({})  # p04 not registered
        assert agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry) is None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_valid_llm_response_returns_plan(self):
        gs = _make_gs()
        router = _FakeModelRouter([(_valid_plan_json(), None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None
        assert plan["fake_seer"] == "p04"
        assert plan["night_kill_primary"] == "p01"
        assert plan["consensus_method"] == "llm"
        assert plan["captain_id"] == "p04"

    def test_discussion_text_injected_into_prompt(self):
        gs = _make_gs()
        router = _FakeModelRouter([(_valid_plan_json(), None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert len(router.calls) == 1
        call_prompt = router.calls[0]["prompt"]
        # Each of 4 wolves has a discussion line
        for w in ("p04", "p05", "p08", "p10"):
            assert f"{w}" in call_prompt
        assert "建议刀 p01" in call_prompt

    def test_alive_wolves_and_targets_in_system_prompt(self):
        gs = _make_gs()
        router = _FakeModelRouter([(_valid_plan_json(), None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        sys_prompt = router.calls[0]["system_prompt"]
        assert "p04" in sys_prompt and "p05" in sys_prompt
        # alive_non_wolves should NOT be in system_prompt (they're in user_prompt)
        # but harder to assert — just check user_prompt
        user_prompt = router.calls[0]["prompt"]
        assert "p01" in user_prompt  # an alive non-wolf
        assert "p06" in user_prompt  # the seer

    def test_prior_plan_carried_over_in_prompt(self):
        gs = _make_gs(night=2)
        router = _FakeModelRouter([(_valid_plan_json(night_number=2), None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        prior = {"fake_seer": "p05", "pusher": "p04", "night_kill_primary": "p06"}
        agent_wolf_team_plan(
            {"game_state": gs, "wolf_team_plan": prior},
            engine=None, registry=registry,
        )
        user_prompt = router.calls[0]["prompt"]
        assert "上夜计划" in user_prompt
        # Either values appear in prompt summary
        assert "p05" in user_prompt


# ---------------------------------------------------------------------------
# Retry / failure handling
# ---------------------------------------------------------------------------

class TestRetryAndFailure:
    def test_repairable_wrapped_payload_is_normalized_before_schema_validation(self):
        gs = _make_gs()
        payload = json.loads(_valid_plan_json())
        payload.pop("public_story")
        payload["reasoning"] = "狼" * 201
        router = _FakeModelRouter([(
            json.dumps({"night_plan": payload}, ensure_ascii=False),
            None,
        )])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})

        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)

        assert plan is not None
        assert plan["night_kill_primary"] == payload["night_kill_primary"]
        assert plan["reasoning"] == payload["reasoning"][:200]
        assert plan["normalization_repairs"] == [
            "unwrap:night_plan",
            "truncate:reasoning",
            "synthesize:public_story",
        ]

    def test_nested_wrapper_remains_schema_invalid(self):
        gs = _make_gs()
        nested = json.dumps({"night_plan": {"night_plan": json.loads(_valid_plan_json())}})
        router = _FakeModelRouter([(nested, None), (nested, None), (nested, None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})

        assert agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry) is None

    def test_invalid_json_triggers_retry_eventually_returns_none(self):
        gs = _make_gs()
        router = _FakeModelRouter([
            ("not json at all", None),
            ("still {not} json", None),
            ("definitely not", None),
        ])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is None
        assert len(router.calls) == 3  # exhausted retries

    def test_schema_invalid_then_valid_succeeds(self):
        gs = _make_gs()
        # First response: missing required field 'reasoning' → schema fail
        bad = json.dumps({
            "night_number": 1, "night_kill_primary": "p01", "night_kill_backup": None,
            "fake_seer": "p04", "pusher": "p05", "hooker": "p08", "deep_cover": "p10",
            "public_story": "go", "evidence_quality": "weak",
        })
        router = _FakeModelRouter([
            (bad, None),
            (_valid_plan_json(), None),
        ])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None
        assert len(router.calls) == 2

    def test_missing_night_number_is_autofilled_from_game_state(self):
        gs = _make_gs(night=2)
        payload = json.loads(_valid_plan_json(night_number=2))
        payload.pop("night_number")
        router = _FakeModelRouter([(json.dumps(payload, ensure_ascii=False), None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})

        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)

        assert plan is not None
        assert plan["night_number"] == 2
        assert len(router.calls) == 1

    def test_duplicate_role_rejected_by_schema(self):
        gs = _make_gs()
        # fake_seer == pusher → schema validator rejects
        dup = _valid_plan_json(fake_seer="p04", pusher="p04")
        router = _FakeModelRouter([(dup, None), (dup, None), (dup, None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is None

    def test_target_not_in_alive_non_wolves_rejected(self):
        gs = _make_gs()
        # night_kill_primary = p99 (does not exist)
        bad = _valid_plan_json(night_kill_primary="p99")
        good = _valid_plan_json()
        router = _FakeModelRouter([(bad, None), (good, None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None
        # First was rejected for membership, second passed
        assert len(router.calls) == 2

    def test_role_pointing_at_dead_wolf_rejected(self):
        gs = _make_gs(wolves=("p04", "p05", "p08", "p10"))
        # Kill p08
        gs = replace(gs, players={
            **gs.players,
            "p08": replace(gs.players["p08"], alive=False),
        })
        # plan still names p08 as hooker — should be rejected by membership check
        bad = _valid_plan_json(hooker="p08")
        good = _valid_plan_json(hooker=None)
        router = _FakeModelRouter([(bad, None), (good, None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None

    def test_provider_not_implemented_switches_remaining_attempts_to_plain_json(self):
        gs = _make_gs()
        state = {"game_state": gs}
        router = _FakeModelRouter([
            ("", NotImplementedError("provider does not support tool_choice")),
            ("", None),
            ("", None),
            ("", None),
        ])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})

        plan = agent_wolf_team_plan(state, engine=None, registry=registry)

        assert plan is None
        assert state["wolf_team_plan_failure"]["reason"] == "empty_response"
        assert router.calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit_wolf_team_plan",
        }
        assert all(call["tool_choice"] is None for call in router.calls[1:])

    def test_provider_without_tool_choice_falls_back_to_plain_json(self):
        gs = _make_gs()
        router = _FakeModelRouter([
            ("", NotImplementedError("provider does not support tool_choice")),
            (_valid_plan_json(), None),
        ])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})

        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)

        assert plan is not None
        assert plan["consensus_method"] == "llm"
        assert len(router.calls) == 2
        assert router.calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit_wolf_team_plan",
        }
        assert router.calls[1]["tool_choice"] is None
        assert "只输出一个完整JSON对象" in router.calls[1]["prompt"]

    def test_empty_response_retries(self):
        gs = _make_gs()
        router = _FakeModelRouter([
            ("", None), ("", None), (_valid_plan_json(), None),
        ])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        plan = agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        assert plan is not None
        assert len(router.calls) == 3

    def test_empty_response_exhaustion_records_failure_metadata(self):
        gs = _make_gs()
        state = {"game_state": gs}
        router = _FakeModelRouter([("", None), ("", None), ("", None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})

        plan = agent_wolf_team_plan(state, engine=None, registry=registry)

        assert plan is None
        assert state["wolf_team_plan_failure"] == {
            "reason": "empty_response",
            "stage": "model_output",
            "attempts": 3,
            "last_error": "empty_response",
            "captain_id": "p04",
        }


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

class TestExtractFirstBalancedJsonObject:
    def test_simple_object(self):
        out = _extract_first_balanced_json_object('{"a": 1}')
        assert out == {"a": 1}

    def test_object_after_text(self):
        out = _extract_first_balanced_json_object(
            'Here is the plan: {"fake_seer": "p04"} thanks'
        )
        assert out == {"fake_seer": "p04"}

    def test_nested_object(self):
        out = _extract_first_balanced_json_object(
            '{"outer": {"inner": 1}, "x": 2}'
        )
        assert out == {"outer": {"inner": 1}, "x": 2}

    def test_no_object_returns_none(self):
        assert _extract_first_balanced_json_object("just plain text") is None

    def test_unterminated_object_returns_none(self):
        assert _extract_first_balanced_json_object('{"a": 1') is None

    def test_braces_in_string_not_counted(self):
        out = _extract_first_balanced_json_object('{"msg": "has } in str", "ok": 1}')
        assert out == {"msg": "has } in str", "ok": 1}


# ---------------------------------------------------------------------------
# Wolf discussion text collection (only current night)
# ---------------------------------------------------------------------------

class TestDiscussionCollection:
    def test_only_current_night_text_collected(self):
        # Add a stale night-2 discussion event to a night-2 game with prior night-1 events
        gs = _make_gs(night=2)
        # Insert prior-night noise
        prior_noise = GameEvent(
            type="wolf_discussion",
            payload={"wolf_id": "p04", "round": 1, "night_number": 1,
                     "text": "OLD NIGHT TEXT SHOULD NOT APPEAR",
                     "visibility": "werewolf_team_only"},
        )
        gs = replace(gs, events=[prior_noise] + list(gs.events))
        router = _FakeModelRouter([(_valid_plan_json(night_number=2), None)])
        registry = _FakeRegistry({"p04": _FakeAgent("p04", router)})
        agent_wolf_team_plan({"game_state": gs}, engine=None, registry=registry)
        user_prompt = router.calls[0]["prompt"]
        assert "OLD NIGHT TEXT" not in user_prompt
        assert "建议刀 p01" in user_prompt  # current night text present
