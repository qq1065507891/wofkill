"""Tests for the extracted trace_builder module."""

from __future__ import annotations

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    PlayerAction,
    PrivateIntent,
    RetryInfo,
    TaskType,
)
from werewolf_agent.agents.trace_builder import build_action_trace


def _context(**overrides) -> AgentContext:
    base: dict = {
        "agent_id": "p01",
        "task_type": TaskType.VOTE,
        "legal_actions": [ActionType.VOTE],
        "legal_targets": ["p07", "p08"],
    }
    base.update(overrides)
    return AgentContext(**base)


class TestBuildActionTrace:
    def test_basic_trace(self):
        retry = RetryInfo(attempt=1, max_retries=3, error_code=None)
        trace = build_action_trace(
            _context(),
            raw_text='{"action_type": "vote"}',
            parsed_action=None,
            final_action_type=ActionType.VOTE,
            retry=retry,
            parse_success=True,
        )
        assert trace.raw_text == '{"action_type": "vote"}'
        assert trace.final_action_type == "vote"
        assert trace.retry == retry.model_dump()
        assert trace.retry_count == 0
        assert trace.fallback_target_used is False
        assert trace.parse_success is True
        assert trace.parse_error is None
        # legal_actions / legal_targets are sourced from the context
        assert trace.legal_actions == ["vote"]
        assert trace.legal_targets == ["p07", "p08"]

    def test_fallback_trace_marks_target(self):
        trace = build_action_trace(
            _context(),
            raw_text="",
            parsed_action=None,
            final_action_type=ActionType.VOTE,
            retry=None,
            fallback_reason="structure failure",
            fallback_target_used=True,
            fallback_target_id="p07",
        )
        assert trace.fallback_target_used is True
        assert trace.fallback_target_id == "p07"
        assert trace.fallback_reason == "structure failure"
        # None retry serializes as None, not omitted
        assert trace.retry is None

    def test_player_action_serialized_without_trace_loop(self):
        action = PlayerAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            speech="",
            reason="r",
            confidence=0.5,
            suspect_reason="p07发言矛盾",
            not_voting_reason="p08没有证据",
            private_reason="我投p07",
            private_intent=PrivateIntent(
                true_role="villager",
                faction_goal="find_wolves",
                claimed_view="good_player_without_night_info",
            ),
        )
        trace = build_action_trace(
            _context(),
            raw_text="{}",
            parsed_action=action,
            final_action_type=ActionType.VOTE,
            retry=RetryInfo(),
        )
        assert trace.parsed_action is not None
        # The nested trace field should be omitted by model_dump(exclude={"trace"})
        assert "trace" not in trace.parsed_action
        assert trace.parsed_action["action_type"] == "vote"
        assert trace.parsed_action["private_intent"]["true_role"] == "villager"

    def test_legal_actions_and_targets_come_from_context(self):
        ctx = _context(
            legal_actions=[ActionType.VOTE, ActionType.SPEECH],
            legal_targets=["p01", "p02", "p03"],
        )
        trace = build_action_trace(
            ctx,
            raw_text="",
            parsed_action=None,
            final_action_type=ActionType.SPEECH,
            retry=None,
        )
        assert trace.legal_actions == ["vote", "speech"]
        assert trace.legal_targets == ["p01", "p02", "p03"]

    def test_retry_count_and_tool_call_metadata(self):
        trace = build_action_trace(
            _context(),
            raw_text="...",
            parsed_action=None,
            final_action_type=ActionType.VOTE,
            retry=RetryInfo(attempt=2, max_retries=3),
            tool_call_required=True,
            tool_call_received=True,
            retry_count=2,
        )
        assert trace.retry_count == 2
        assert trace.tool_call_required is True
        assert trace.tool_call_received is True
        # tool_call_name is auto-set to "submit_player_action" when required
        assert trace.tool_call_name == "submit_player_action"

    def test_structured_failure_reason_propagates(self):
        trace = build_action_trace(
            _context(),
            raw_text="",
            parsed_action=None,
            final_action_type=ActionType.NO_ACTION,
            retry=RetryInfo(),
            structured_failure_reason="missing_tool_call",
            parse_success=False,
            parse_error="missing required tool call",
        )
        assert trace.structured_failure_reason == "missing_tool_call"
        assert trace.parse_success is False
        assert trace.parse_error == "missing required tool call"

    def test_dict_parsed_action_passes_through(self):
        # When parsed_action is a plain dict (e.g. choice_data from
        # parse_choice_action), the builder should accept it directly
        # without trying to call model_dump on it.
        choice_data = {
            "target_id": "p07",
            "reason": "r",
            "confidence": 0.7,
            "seer_stance": "undecided",
            "vote_basis": "fallback",
            "standing_with_seer": "",
            "suspect_reason": "",
            "not_voting_reason": "",
            "private_reason": "",
        }
        trace = build_action_trace(
            _context(),
            raw_text='{"choice": "a"}',
            parsed_action=choice_data,
            final_action_type=ActionType.VOTE,
            retry=RetryInfo(),
        )
        # Pydantic v2 re-validates the dict into a fresh mapping, so we
        # compare by value rather than identity.
        assert trace.parsed_action == choice_data
        assert trace.parsed_action["target_id"] == "p07"
