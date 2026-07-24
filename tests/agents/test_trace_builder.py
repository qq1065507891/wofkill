# -*- coding: utf-8 -*-
"""
验证 ActionTrace 构造、V2 计数与终态失败码投影。

作者: Project contributors
修改日期: 2026-07-24
"""

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
from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    EvidenceKind,
    OpaqueRequestId,
    ReasoningLevel,
    ReasoningStatus,
    RootCause,
    RouteKind,
)


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
    def test_execution_attempts_derive_runtime_timeout_count(self):
        request_id = OpaqueRequestId.new("game", "11223344")
        attempts = (
            AttemptExecutionRecord(
                opaque_request_id=request_id,
                ordinal=1,
                provider="primary",
                model="model-a",
                route_kind=RouteKind.PRIMARY,
                root_cause=RootCause.TIMEOUT,
                attempt_outcome=AttemptOutcome.FAILURE,
                requested_reasoning_level=ReasoningLevel.HIGH,
                normalized_reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
                reasoning_token_count=0,
                evidence_kind=EvidenceKind.NONE,
            ),
            AttemptExecutionRecord(
                opaque_request_id=request_id,
                ordinal=2,
                provider="primary",
                model="model-a",
                route_kind=RouteKind.RETRY,
                root_cause=RootCause.NONE,
                attempt_outcome=AttemptOutcome.SUCCESS,
                requested_reasoning_level=ReasoningLevel.HIGH,
                normalized_reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
                reasoning_token_count=0,
                evidence_kind=EvidenceKind.NONE,
            ),
        )

        trace = build_action_trace(
            _context(), raw_text="", parsed_action=None,
            final_action_type=ActionType.NO_ACTION, retry=RetryInfo(),
            execution_attempts=attempts,
        )

        assert trace.runtime_timeout_count == 1
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
        expected_retry = retry.model_dump(exclude={"correction_hint"})
        assert trace.retry == expected_retry
        assert trace.retry_count == 0
        assert trace.fallback_target_used is False
        assert trace.parse_success is True
        assert trace.parse_error is None
        # legal_actions / legal_targets are sourced from the context
        assert trace.legal_actions == ["vote"]
        assert trace.legal_targets == ["p07", "p08"]

    def test_retry_correction_hint_is_redacted_but_audit_fields_remain(self):
        """构造 trace 时只移除模型修正提示，保留其他稳定审计字段。"""
        sentinel = "CORRECTION_HINT_SENTINEL"
        retry = RetryInfo(
            attempt=2,
            max_retries=3,
            error_code="speech_quality",
            error_message="缺少明确论点",
            reason_codes=["unsupported_public_claim"],
            correction_hint=sentinel,
            early_exit_reason="repeat_error_signature",
            failure_category="unknown",
        )

        trace = build_action_trace(
            _context(),
            raw_text="{}",
            parsed_action=None,
            final_action_type=ActionType.VOTE,
            retry=retry,
        )

        assert retry.correction_hint == sentinel
        assert trace.retry == {
            "attempt": 2,
            "max_retries": 3,
            "error_code": "speech_quality",
            "error_message": "缺少明确论点",
            "reason_codes": ["unsupported_public_claim"],
            "early_exit_reason": "repeat_error_signature",
            "failure_category": "unknown",
        }
        assert sentinel not in trace.model_dump_json()

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

    def test_terminal_failure_code_matches_normalized_trace(self):
        from werewolf_agent.model_gateway.execution_records import (
            AttemptExecutionRecord,
            AttemptOutcome,
            EvidenceKind,
            OpaqueRequestId,
            ReasoningLevel,
            ReasoningStatus,
            RootCause,
            RouteKind,
        )
        from werewolf_agent.runtime.decision_outcomes import (
            normalize_decision_execution_trace,
        )

        request_id = OpaqueRequestId.new("game", "11223344")
        common = dict(
            opaque_request_id=request_id,
            provider="primary",
            model="model-a",
            requested_reasoning_level=ReasoningLevel.HIGH,
            reasoning_token_count=0,
        )
        attempts = (
            AttemptExecutionRecord(
                **common,
                ordinal=1,
                route_kind=RouteKind.PRIMARY,
                root_cause=RootCause.INVALID_OUTPUT,
                attempt_outcome=AttemptOutcome.FAILURE,
                normalized_reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
                evidence_kind=EvidenceKind.NONE,
            ),
            AttemptExecutionRecord(
                **common,
                ordinal=2,
                route_kind=RouteKind.SAFE_FALLBACK,
                root_cause=RootCause.INVALID_OUTPUT,
                attempt_outcome=AttemptOutcome.FAILURE,
                normalized_reasoning_status=ReasoningStatus.FALLBACK_DISABLED,
                evidence_kind=EvidenceKind.FALLBACK_DISABLED,
            ),
        )
        trace = build_action_trace(
            _context(),
            raw_text="",
            parsed_action=None,
            final_action_type=ActionType.NO_ACTION,
            retry=RetryInfo(),
            structured_failure_reason="illegal_action",
            execution_attempts=attempts,
        )

        normalized = normalize_decision_execution_trace(trace.model_dump())

        assert trace.terminal_failure_code == "illegal_action"
        assert normalized["terminal_failure_code"] == "illegal_action"

    def test_structured_output_mode_and_failure_stage_propagate(self):
        trace = build_action_trace(
            _context(),
            raw_text="not json",
            parsed_action=None,
            final_action_type=ActionType.NO_ACTION,
            retry=RetryInfo(),
            structured_output_mode="json_schema",
            structured_failure_stage="protocol",
        )

        assert trace.structured_output_mode == "json_schema"
        assert trace.structured_failure_stage == "protocol"

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

    def test_world_model_audit_attaches_sanitized_context_and_plan(self):
        ctx = _context(
            belief_state={
                "my_suspects": [
                    {
                        "player": "p07",
                        "top_role_guess": "werewolf",
                        "top_role_prob": 0.9,
                    }
                ]
            },
            possible_worlds={
                "top_worlds": [
                    {
                        "label": "World A",
                        "roles": {"p07": "werewolf"},
                        "key_assignments": {"p07": "werewolf"},
                    }
                ]
            },
            simulation_predictions={
                "predictions": [
                    {
                        "event": "next_day_vote_pressure",
                        "affected_players": ["p07"],
                    }
                ]
            },
        )
        parsed_action = {
            "planning_mode": "decision_dialogue",
            "decision_plan": {
                "action_type": "vote",
                "target_id": "p07",
                "private_goal": "protect p08",
            },
            "dialogue_plan": {
                "public_intent": "push p07",
                "talking_points": ["p07 changed stance twice"],
                "conceal": ["p08 is my wolf teammate"],
            },
            "persona_policy_prior": {"vote_threshold": 0.62},
        }

        trace = build_action_trace(
            ctx,
            raw_text="{}",
            parsed_action=parsed_action,
            final_action_type=ActionType.VOTE,
            retry=RetryInfo(),
        )

        audit = trace.world_model_audit
        assert audit["player_id"] == "p01"
        assert audit["belief"]["my_suspects"][0]["player"] == "p07"
        assert audit["possible_worlds"]["top_worlds"][0]["key_assignments"] == {
            "p07": "werewolf"
        }
        assert audit["simulation_predictions"]["predictions"][0]["affected_players"] == ["p07"]
        assert audit["decision_plan"]["action_type"] == "vote"
        assert audit["dialogue_plan"]["public_intent"] == "push p07"
        assert audit["persona_policy_prior"]["vote_threshold"] == 0.62
        assert "private_goal" not in str(audit)
        assert "conceal" not in str(audit)
        assert "roles" not in str(audit)
