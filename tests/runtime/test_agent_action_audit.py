# -*- coding: utf-8 -*-
"""
验证 agent action audit 辅助函数拆分后的兼容行为。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-23

使用示例:
    >>> python -m pytest tests/runtime/test_agent_action_audit.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
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


def _timeout_trace_payload() -> dict[str, object]:
    from werewolf_agent.agents.trace_builder import build_action_trace

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
        AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02"],
        ),
        raw_text="",
        parsed_action=None,
        final_action_type=ActionType.NO_ACTION,
        retry=RetryInfo(),
        execution_attempts=attempts,
    )
    return trace.model_dump()


def test_action_audit_event_always_carries_explicit_task_type() -> None:
    from werewolf_agent.runtime.nodes.action_audit import _action_trace_event

    event = _action_trace_event(
        player_id="p01",
        phase="vote",
        action_trace={"execution_attempts": []},
    )

    assert event.payload["task_type"] == "vote"


def test_action_audit_keeps_runtime_timeout_count_moderator_only() -> None:
    from werewolf_agent.runtime.nodes.action_audit import _action_trace_event

    event = _action_trace_event(
        player_id="p01",
        phase="vote",
        action_trace=_timeout_trace_payload(),
    )

    assert event.payload["visibility"] == "moderator_only"
    assert event.payload["action_trace"]["runtime_timeout_count"] == 1


def test_action_audit_emits_separate_moderator_only_semantic_repair_event() -> None:
    from werewolf_agent.runtime.nodes.action_audit import _action_audit_events

    semantic = {
        "repairable": True,
        "success": True,
        "target_preserved": True,
        "introduced_claim_count": 0,
        "verified_claim_count": 1,
        "retained_verified_claim_count": 1,
        "generic_template_used": False,
    }
    events = _action_audit_events(
        state={},
        player_id="p01",
        phase="speech",
        action_trace={"semantic_repair_audit": semantic},
        decision_identity=None,
        exposure_collector=None,
    )

    assert [event.type for event in events] == [
        "semantic_repair_audit", "action_trace_audit"
    ]
    assert events[0].payload == {**semantic, "visibility": "moderator_only"}


def test_semantic_repair_event_carries_decision_identity() -> None:
    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.runtime.nodes.action_audit import _action_audit_events

    identity = DecisionIdentity(
        game_id="g1", player_id="p01", phase="day", day_number=1,
        night_number=0, task_type="speech", action_index=4,
    )
    events = _action_audit_events(
        state={}, player_id="p01", phase="speech",
        action_trace={"semantic_repair_audit": {"repairable": True}},
        decision_identity=identity, exposure_collector=None,
    )

    semantic = events[0].payload
    assert semantic["trace_id"] == identity.trace_id()
    assert semantic["game_id"] == "g1"
    assert semantic["action_index"] == 4
    assert semantic["task_type"] == "speech"


def test_audit_context_kwargs_requires_identity_and_collector() -> None:
    from werewolf_agent.runtime.agent_action_audit import _audit_context_kwargs

    sink = object()

    assert _audit_context_kwargs(None, object(), sink) == {}
    assert _audit_context_kwargs(object(), None, sink) == {}

    identity = object()
    collector = object()
    result = _audit_context_kwargs(identity, collector, sink)

    assert result == {
        "decision_identity": identity,
        "exposure_collector": collector,
        "decision_trace_sink": sink,
    }


def test_vote_basis_hint_skips_seer_and_applies_to_non_seer() -> None:
    from werewolf_agent.runtime.agent_action_audit import (
        VOTE_BASIS_GUIDANCE,
        _inject_vote_basis_hint,
    )

    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="villager"),
        },
    )

    seer_directive: dict[str, object] = {}
    villager_directive: dict[str, object] = {}

    _inject_vote_basis_hint(seer_directive, gs, "p01")
    _inject_vote_basis_hint(villager_directive, gs, "p02")

    assert "vote_basis_hint" not in seer_directive
    assert villager_directive["vote_basis_hint"] == VOTE_BASIS_GUIDANCE


def test_seer_credibility_audit_payload_keeps_safe_public_fields() -> None:
    from werewolf_agent.runtime.agent_action_audit import _seer_credibility_audit_payload

    context = AgentContext(
        agent_id="p03",
        task_type=TaskType.SPEECH,
        seer_credibility={
            "seer_lines": [
                {
                    "claimant": "p01",
                    "status": "credible",
                    "score": 0.8,
                    "hidden_role": "seer",
                },
            ],
        },
    )

    payload = _seer_credibility_audit_payload(context, day_number=2)

    assert payload == {
        "day_number": 2,
        "visibility": "moderator_only",
        "seer_lines": [{"claimant": "p01", "status": "credible", "score": 0.8}],
    }


def test_is_sheriff_silenced_uses_event_or_badge_state() -> None:
    from werewolf_agent.runtime.agent_action_audit import _is_sheriff_silenced

    gs = GameState(
        sheriff_badge_state="active",
        events=[GameEvent(type="sheriff_silenced", payload={"sheriff_id": "p01"})],
    )
    assert _is_sheriff_silenced(gs, "p01") is True
    assert _is_sheriff_silenced(gs, "p02") is False

    frozen = GameState(sheriff_badge_state="frozen")
    assert _is_sheriff_silenced(frozen, "p03") is True
