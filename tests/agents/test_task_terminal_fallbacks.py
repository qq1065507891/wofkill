# -*- coding: utf-8 -*-
"""
验证不同任务的终端回退内容、失败归因和 V2 审计字段。

作者: Project contributors
创建日期: 2026-07-16
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.player_fallback_speech import (
    build_task_terminal_fallback,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
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


def _context(
    task_type: TaskType,
    *,
    legal_actions: list[ActionType] | None = None,
    legal_targets: list[str] | None = None,
) -> AgentContext:
    return AgentContext(
        agent_id="p01",
        task_type=task_type,
        phase="night" if task_type is TaskType.NIGHT_ACTION else "day",
        day_number=2,
        night_number=1,
        own_role="villager",
        legal_actions=legal_actions or [ActionType.SPEECH],
        legal_targets=legal_targets or [],
        visible_world_state={"alive_players": ["p01", "p02", "p03"]},
        public_summary="公开信息：昨夜无人出局。",
    )


@pytest.mark.parametrize(
    ("task_type", "expected_kind", "expects_text"),
    [
        (TaskType.SPEECH, "ordinary_speech", True),
        (TaskType.SHERIFF_SPEECH, "sheriff_speech", True),
        (TaskType.REFLECTION, "reflection_not_generated", False),
        (TaskType.LAST_WORDS, "last_words_not_generated", False),
        (TaskType.WOLF_TEAM_PLAN, "wolf_team_plan_structured_stance", False),
    ],
)
def test_terminal_fallback_is_task_specific(
    task_type: TaskType,
    expected_kind: str,
    expects_text: bool,
) -> None:
    context = _context(task_type)
    base = FallbackAction(action_type=ActionType.SPEECH, reason="generic failure")

    action, fallback_kind = build_task_terminal_fallback(context, base)

    assert fallback_kind == expected_kind
    assert bool(action.speech.strip()) is expects_text
    if task_type in {TaskType.REFLECTION, TaskType.LAST_WORDS, TaskType.WOLF_TEAM_PLAN}:
        assert action.action_type is ActionType.NO_ACTION
        assert action.target_id is None
    if task_type is TaskType.REFLECTION:
        assert action.reason == "not_generated"


def test_night_terminal_fallback_selects_only_deterministic_legal_action() -> None:
    context = _context(
        TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT],
        legal_targets=["p03", "p02"],
    )

    action, fallback_kind = build_task_terminal_fallback(
        context,
        FallbackAction(action_type=ActionType.NO_ACTION),
    )

    assert fallback_kind == "night_legal_action"
    assert action.action_type is ActionType.CHECK_ALIGNMENT
    assert action.target_id == "p03"
    assert action.speech == ""


def test_night_terminal_fallback_abstains_when_no_legal_target_exists() -> None:
    context = _context(
        TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT],
        legal_targets=[],
    )

    action, fallback_kind = build_task_terminal_fallback(
        context,
        FallbackAction(action_type=ActionType.CHECK_ALIGNMENT),
    )

    assert fallback_kind == "night_explicit_abstain"
    assert action.action_type is ActionType.NO_ACTION
    assert action.target_id is None
    assert action.reason == "no_legal_deterministic_action"


def _terminal_attempts() -> tuple[AttemptExecutionRecord, ...]:
    request_id = OpaqueRequestId.new("game", "a11f00ba")
    common = {
        "opaque_request_id": request_id,
        "provider": "provider-a",
        "model": "model-a",
        "requested_reasoning_level": ReasoningLevel.HIGH,
        "normalized_reasoning_status": ReasoningStatus.REQUESTED_UNCONFIRMED,
        "reasoning_token_count": 0,
        "evidence_kind": EvidenceKind.NONE,
        "attempt_outcome": AttemptOutcome.FAILURE,
    }
    return (
        AttemptExecutionRecord(
            ordinal=1,
            route_kind=RouteKind.PRIMARY,
            root_cause=RootCause.INVALID_OUTPUT,
            **common,
        ),
        AttemptExecutionRecord(
            ordinal=2,
            route_kind=RouteKind.SAFE_FALLBACK,
            root_cause=RootCause.POLICY_REJECTION,
            **common,
        ),
    )


def test_terminal_trace_preserves_stable_failure_without_raw_error_text() -> None:
    trace = build_action_trace(
        _context(TaskType.SPEECH),
        raw_text="failed model text, not an exception body",
        parsed_action=None,
        final_action_type=ActionType.SPEECH,
        retry=RetryInfo(
            error_code="schema_validation",
            error_message="private provider exception body",
        ),
        parse_error="private provider exception body",
        structured_failure_reason="schema_validation",
        structured_failure_stage="protocol",
        execution_attempts=_terminal_attempts(),
        fallback_kind="ordinary_speech",
    )

    payload = trace.model_dump()
    assert payload["generated_by"] == "terminal_fallback"
    assert payload["terminal_failure_code"] == "schema_validation"
    assert payload["original_failure_code"] == "schema_validation"
    assert payload["failure_stage"] == "protocol"
    assert payload["fallback_kind"] == "ordinary_speech"
    assert "private provider exception body" not in trace.model_dump_json()


def test_terminal_trace_unknown_failure_code_fails_closed() -> None:
    trace = build_action_trace(
        _context(TaskType.SPEECH),
        raw_text="",
        parsed_action=None,
        final_action_type=ActionType.SPEECH,
        retry=RetryInfo(error_code="secret-player-role"),
        structured_failure_reason="secret-player-role",
        structured_failure_stage="semantic",
        execution_attempts=_terminal_attempts(),
        fallback_kind="ordinary_speech",
    )

    assert trace.terminal_failure_code == "unknown"
    assert trace.original_failure_code == "unknown"
