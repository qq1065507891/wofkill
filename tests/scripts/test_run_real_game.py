# -*- coding: utf-8 -*-
"""
验证真实游戏脚本的报告辅助函数与结构化质量指标。

作者: Project contributors
修改日期: 2026-07-18
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.run_real_game import _safe_event_payload, print_quality_audit
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.core.resolution_batches import ResolutionBatchV2


def _save_game_log(run_real_game, runner, elapsed, **kwargs):
    """测试新写链路：先固定最终投影和质量，再交给纯保存函数。"""
    projection = run_real_game.project_acceptance_game(
        runner.state,
        steps=runner.step_count,
    )
    quality_score = kwargs.pop("quality_score", None)
    if quality_score is None:
        quality_score = run_real_game.compute_game_quality_score(projection)
    return run_real_game.save_game_log(
        runner,
        elapsed,
        projection=projection,
        quality_score=quality_score,
        **kwargs,
    )


def test_runner_config_routes_cli_output_dir_to_emergency_artifacts(tmp_path) -> None:
    from scripts import run_real_game

    args = run_real_game._build_argument_parser().parse_args([
        "--seed", "42", "--output-dir", str(tmp_path),
    ])
    config = run_real_game._build_runner_config(
        args, game_repo=None, memory_coordinator=None,
    )

    assert config.emergency_artifact_dir == tmp_path


def test_terminal_log_message_distinguishes_finished_and_aborted(caplog) -> None:
    from scripts import run_real_game

    finished = SimpleNamespace(
        state=GameState(
            game_id="g-finished", phase="finished", status="finished",
            winning_faction="good",
        ),
        step_count=7,
    )
    aborted = SimpleNamespace(
        state=GameState(
            game_id="g-aborted", phase="night", status="aborted",
            termination_reason="step_limit",
        ),
        step_count=50,
    )

    with caplog.at_level("INFO"):
        assert run_real_game.log_terminal_outcome(finished, 1.5, {"fallback_rate": 0.0}) == 0
    assert "GAME_COMPLETE winner=good" in caplog.text

    caplog.clear()
    with caplog.at_level("INFO"):
        assert run_real_game.log_terminal_outcome(aborted, 2.5, {"fallback_rate": 0.0}) == 1

    assert "GAME_ABORTED reason=step_limit" in caplog.text
    assert "GAME_COMPLETE" not in caplog.text


def test_finalize_game_log_projects_after_persistence_audit_and_scores_once(
    monkeypatch,
) -> None:
    from scripts import run_real_game

    runner = SimpleNamespace(
        game_id="g-final-order",
        state=GameState(
            game_id="g-final-order",
            players={"p01": PlayerState(id="p01", role="villager")},
            events=[GameEvent(type="reflection_persistence_audit", payload={})],
        ),
        step_count=3,
    )
    calls: list[str] = []
    expected_projection = object()
    quality = {"fallback_rate": 0.0, "total_quality_events": 0}

    def project(state, *, steps):
        assert state.events[-1].type == "reflection_persistence_audit"
        assert steps == 3
        calls.append("project")
        return expected_projection

    def compute(value):
        assert value is expected_projection
        calls.append("compute")
        return quality

    def save(_runner, _elapsed, *, projection, quality_score, output_dir):
        assert projection is expected_projection
        assert quality_score is quality
        assert output_dir is None
        calls.append("save")
        return "game.json"

    monkeypatch.setattr(run_real_game, "project_acceptance_game", project)
    monkeypatch.setattr(run_real_game, "compute_game_quality_score", compute)
    monkeypatch.setattr(run_real_game, "save_game_log", save)

    path, returned_quality = run_real_game.finalize_game_log(runner, 1.0)

    assert path == "game.json"
    assert returned_quality is quality
    assert calls == ["project", "compute", "save"]


def test_quality_score_counts_rejected_reflection_claims_and_lessons_separately() -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g1",
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
        },
        events=[
            GameEvent(type="vote", payload={"voter": "p01", "target": "p02"}),
            GameEvent(type="reflection_complete", payload={"entries": [{
                "player_id": "p01",
                "verification": {
                    "status": "verified",
                    "verified_fact_count": 0,
                    "verified_lessons": [],
                    "rejected_fact_count": 1,
                    "rejected_lesson_count": 1,
                },
            }]}),
        ],
    )

    quality = run_real_game.compute_game_quality_score(SimpleNamespace(state=gs, step_count=1))

    assert quality["reflection_rejected_fact_count"] == 1
    assert quality["reflection_rejected_lesson_count"] == 1


def test_quality_score_exports_persona_confirmation_from_real_events() -> None:
    from scripts import run_real_game

    trace_id = "g1:p01:vote:D1:N0:vote:1"
    gs = GameState(game_id="g1", events=[
        GameEvent(type="persona_exposure_audit", payload={
            "trace_id": trace_id, "snapshot": {"profile_id": "calm"},
        }),
        GameEvent(type="persona_prompt_injection_audit", payload={
            "trace_id": trace_id,
            "proof": {"confirmed_injection": True, "attempt_ordinal": 1},
        }),
    ])

    quality = run_real_game.compute_game_quality_score(SimpleNamespace(state=gs, step_count=1))

    assert quality["persona_prompt_confirmation"] == {
        "supported": True,
        "configured_action_count": 1,
        "confirmed_action_count": 1,
        "confirmation_rate": 1.0,
    }


def test_quality_score_exports_translated_execution_acceptance_metrics() -> None:
    from scripts import run_real_game
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

    request_id = OpaqueRequestId.new("game", "feedbeef")
    attempts = (
        AttemptExecutionRecord(
            opaque_request_id=request_id,
            ordinal=1,
            provider="primary",
            model="m",
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
            model="m",
            route_kind=RouteKind.RETRY,
            root_cause=RootCause.NONE,
            attempt_outcome=AttemptOutcome.SUCCESS,
            requested_reasoning_level=ReasoningLevel.HIGH,
            normalized_reasoning_status=ReasoningStatus.CONFIRMED,
            reasoning_token_count=2,
            evidence_kind=EvidenceKind.TOKEN_COUNT,
        ),
    )
    gs = GameState(game_id="g-execution", events=[GameEvent(
        type="action_trace_audit",
        payload={
            "task_type": "vote",
            "action_trace": {"execution_attempts": attempts},
        },
    )])

    quality = run_real_game.compute_game_quality_score(
        SimpleNamespace(state=gs, step_count=1)
    )

    assert quality["decision_outcome_counts"] == {"retry_success": 1}
    assert quality["attempt_count"] == 2
    assert quality["retry_count"] == 1
    assert quality["reasoning_confirmation_rate"] == 0.5


def test_reflection_metrics_count_only_latest_canonical_decision_per_player() -> None:
    from scripts.run_real_game_reports import reflection_verification_metrics

    def event(decision_id: str, rejected: int) -> GameEvent:
        return GameEvent(type="reflection_complete", payload={"entries": [{
            "player_id": "p01", "decision_id": decision_id,
            "verification": {
                "status": "verified", "decision_id": decision_id,
                "verified_fact_count": 0, "verified_lessons": [],
                "rejected_fact_count": rejected, "rejected_lesson_count": rejected,
            },
        }]})

    gs = GameState(game_id="g-canonical", events=[
        event("d1", 1), event("d1", 1), event("d2", 3),
    ])

    assert reflection_verification_metrics(gs) == {
        "reflection_rejected_fact_count": 3,
        "reflection_rejected_lesson_count": 3,
    }


def test_game_log_reflection_payload_drops_raw_provider_draft() -> None:
    payload = {
        "visibility": "moderator_only",
        "status": "complete",
        "persistence_complete": False,
        "player_count": 1,
        "valid_entry_count": 1,
        "failure_count": 0,
        "entries": [{
            "player_id": "p01",
            "role": "seer",
            "decision_id": "reflection:g1:p01",
            "transaction_state": "lessons_verified",
            "failure_stage": None,
            "failure_code": None,
            "entry_id": None,
            "reflection": "RAW_PROVIDER_DRAFT",
            "provider_response": {"thinking": "SECRET"},
            "private_prompt": "PRIVATE_PROMPT_SECRET",
            "original_text": "ORIGINAL_REFLECTION_SECRET",
            "verification": {
                "status": "verified", "decision_id": "reflection:g1:p01",
                "verified_fact_count": 1,
                "verified_lessons": [{"lesson_id": "l1", "abstraction": "先复核公开票型"}],
                "rejected_fact_count": 0, "rejected_lesson_count": 0,
            },
        }],
    }

    safe = _safe_event_payload(
        "reflection_complete",
        payload,
        game_id="g1",
        players={"p01": {"id": "p01", "role": "seer", "alive": True}},
    )
    serialized = json.dumps(safe, ensure_ascii=False)

    assert "RAW_PROVIDER_DRAFT" not in serialized
    assert "SECRET" not in serialized
    assert "PRIVATE_PROMPT_SECRET" not in serialized
    assert "ORIGINAL_REFLECTION_SECRET" not in serialized
    assert safe["entries"][0]["verification"]["verified_fact_count"] == 1
    assert safe["entries"][0]["decision_id"] == "reflection:g1:p01"
    assert safe["entries"][0]["verification"]["decision_id"] == "reflection:g1:p01"
    assert safe["status"] == "complete"
    assert safe["valid_entry_count"] == 1
    assert safe["failure_count"] == 0
    assert safe["entries"][0]["transaction_state"] == "lessons_verified"


class _UntrustedReflectionValue:
    """任何保存边界上的隐式字符串转换都会使探针立即失败。"""

    def __str__(self) -> str:
        raise AssertionError("untrusted reflection value was coerced with str()")


def test_reflection_sanitizer_never_stringifies_untrusted_nested_values() -> None:
    payload = {
        "status": "partial",
        "player_count": 1,
        "valid_entry_count": 0,
        "failure_count": 1,
        "entries": [{
            "player_id": _UntrustedReflectionValue(),
            "role": _UntrustedReflectionValue(),
            "verification": {"status": "agent_error"},
        }],
    }

    safe = _safe_event_payload(
        "reflection_complete",
        payload,
        game_id="g-reflection-boundary",
        players={"p01": {"id": "p01", "role": "seer", "alive": True}},
    )

    assert safe["entries"] == []


def test_saved_reflection_events_redact_nested_keys_and_marker_values(
    tmp_path,
) -> None:
    """complete/audit 同时重建，脱敏后在线与离线 quality 必须逐字段一致。"""
    from scripts import run_real_game

    game_id = "g-reflection-boundary"
    decision_id = f"reflection:{game_id}:p01"
    claim_ids = ["claim-p01", "raw_prompt"]
    complete = GameEvent(type="reflection_complete", payload={
        "visibility": "moderator_only",
        "status": "complete",
        "persistence_complete": True,
        "player_count": 1,
        "valid_entry_count": 1,
        "failure_count": 0,
        "raw_prompt": {"nested": "provider_response"},
        "entries": [{
            "player_id": "p01",
            "role": "seer",
            "alive": True,
            "decision_id": decision_id,
            "transaction_state": "lessons_verified",
            "failure_stage": None,
            "failure_code": None,
            "entry_id": None,
            "private_prompt": {"carrier": "original_text"},
            "verification": {
                "status": "verified",
                "decision_id": decision_id,
                "verified_fact_count": 2,
                "verified_claim_ids": claim_ids,
                "rejected_claim_ids": ["provider_response"],
                "verified_lessons": [{
                    "lesson_id": "original_text",
                    "abstraction": "PRIVATE_PROMPT secret tactical prose",
                    "provider_response": "must disappear",
                }],
                "rejected_fact_count": 0,
                "rejected_lesson_count": 0,
            },
        }],
    })
    persistence = GameEvent(type="reflection_persistence_audit", payload={
        "status": "complete",
        "expected_entry_count": 1,
        "persistence_complete": True,
        "rollback_complete": True,
        "original_text": {"raw_prompt": "must disappear"},
        "entries": [{
            "player_id": "p01",
            "decision_id": decision_id,
            "verified_claim_ids": claim_ids,
            "entry_id": f"reflection_{game_id}_p01",
            "row_found": True,
            "persistence_complete": True,
            "persisted_rejected_fact_count": 0,
            "provider_response": {"private_prompt": "must disappear"},
        }],
    })
    runner = SimpleNamespace(
        game_id=game_id,
        state=GameState(
            game_id=game_id,
            phase="finished",
            status="finished",
            winning_faction="good",
            players={"p01": PlayerState(id="p01", role="seer", alive=True)},
            events=[complete, persistence],
        ),
        step_count=2,
    )
    projection = run_real_game.project_acceptance_game(
        runner.state,
        steps=runner.step_count,
    )
    saved_quality = run_real_game.compute_game_quality_score(projection)

    path = run_real_game.save_game_log(
        runner,
        elapsed=0.1,
        projection=projection,
        quality_score=saved_quality,
        output_dir=tmp_path,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(saved["events"], ensure_ascii=False).lower()
    offline_quality = run_real_game.compute_game_quality_score(
        run_real_game.project_acceptance_game(saved)
    )

    for marker in (
        "raw_prompt", "provider_response", "private_prompt", "original_text",
        "secret tactical prose", "claim-p01",
    ):
        assert marker not in serialized
    complete_entry = saved["events"][0]["payload"]["entries"][0]
    audit_entry = saved["events"][1]["payload"]["entries"][0]
    safe_claim_ids = complete_entry["verification"]["verified_claim_ids"]
    assert safe_claim_ids == audit_entry["verified_claim_ids"]
    assert all(item.startswith("redacted_claim_") for item in safe_claim_ids)
    lesson = complete_entry["verification"]["verified_lessons"][0]
    assert lesson["lesson_id"].startswith("redacted_lesson_")
    assert lesson["abstraction"] == "[REDACTED_VERIFIED_LESSON]"
    assert saved["quality_score"] == saved_quality
    assert offline_quality == saved_quality


def test_reasoning_evidence_summary_is_allowlisted_and_has_exact_denominators():
    from werewolf_agent.model_gateway.execution_records import (
        AttemptExecutionRecord, AttemptOutcome, EvidenceKind, OpaqueRequestId,
        ReasoningLevel, ReasoningStatus, RootCause, RouteKind,
    )
    from werewolf_agent.model_gateway.usage_records import UsageRecord
    from scripts.run_real_game_reports import _reasoning_evidence_summary

    attempt = AttemptExecutionRecord(
        opaque_request_id=OpaqueRequestId.new("game", "abcdef12"), ordinal=1,
        provider="openai", model="reasoner", route_kind=RouteKind.PRIMARY,
        root_cause=RootCause.NONE, attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.CONFIRMED,
        reasoning_token_count=4, evidence_kind=EvidenceKind.TOKEN_COUNT,
    )
    summary = _reasoning_evidence_summary([
        UsageRecord(agent_id="p01", task_type="reflection", provider="openai", model="reasoner", attempts=(attempt,)),
    ])

    assert summary["requested_denominator"] == 1
    assert summary["confirmed_numerator"] == 1
    assert summary["support_flags"] == {"reasoning_token_evidence": True, "provider_status_evidence": False}
    assert set(summary["attempts"][0]) == {
        "opaque_request_id", "ordinal", "provider", "model", "requested_level",
        "status", "reasoning_tokens", "evidence", "route", "root_cause", "outcome",
    }


def test_reasoning_summary_canonicalizes_snapshots_and_prefers_action_projection():
    from dataclasses import replace
    from werewolf_agent.model_gateway.execution_records import (
        AttemptExecutionRecord, AttemptOutcome, EvidenceKind, OpaqueRequestId,
        ReasoningLevel, ReasoningStatus, RootCause, RouteKind,
    )
    from werewolf_agent.model_gateway.usage_records import UsageRecord
    from scripts.run_real_game_reports import _reasoning_evidence_summary

    request_id = OpaqueRequestId.new("game", "1234abcd")
    success = AttemptExecutionRecord(
        opaque_request_id=request_id, ordinal=1, provider="openai", model="m",
        route_kind=RouteKind.PRIMARY, root_cause=RootCause.NONE,
        attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.CONFIRMED,
        reasoning_token_count=2, evidence_kind=EvidenceKind.TOKEN_COUNT,
    )
    repaired = replace(success, root_cause=RootCause.INVALID_OUTPUT, attempt_outcome=AttemptOutcome.FAILURE)
    final = replace(success, ordinal=2, route_kind=RouteKind.REPAIR)
    usage = [
        UsageRecord(agent_id="p01", task_type="vote", provider="openai", model="m", attempts=(success,)),
        UsageRecord(agent_id="p01", task_type="vote", provider="openai", model="m", attempts=(success, final)),
    ]
    summary = _reasoning_evidence_summary(usage, action_attempts=(repaired, final))
    assert summary["requested_denominator"] == 2
    assert len(summary["attempts"]) == 2
    assert summary["attempts"][0]["root_cause"] == "invalid_output"
    assert summary["attempts"][0]["outcome"] == "attempt_failure"


def test_reasoning_summary_groups_interleaved_requests_by_first_seen_order():
    from dataclasses import replace
    from werewolf_agent.model_gateway.execution_records import (
        AttemptExecutionRecord, AttemptOutcome, EvidenceKind, OpaqueRequestId,
        ReasoningLevel, ReasoningStatus, RootCause, RouteKind,
    )
    from werewolf_agent.model_gateway.usage_records import UsageRecord
    from scripts.run_real_game_reports import _reasoning_evidence_summary

    request_a = OpaqueRequestId.new("game", "aaaabbbb")
    request_b = OpaqueRequestId.new("game", "ccccdddd")
    a1 = AttemptExecutionRecord(
        opaque_request_id=request_a, ordinal=1, provider="a", model="m",
        route_kind=RouteKind.PRIMARY, root_cause=RootCause.NONE,
        attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.CONFIRMED,
        reasoning_token_count=1, evidence_kind=EvidenceKind.TOKEN_COUNT,
    )
    b1 = replace(a1, opaque_request_id=request_b, provider="b")
    a2 = replace(a1, ordinal=2, route_kind=RouteKind.REPAIR)
    usage = [
        UsageRecord(agent_id="p01", task_type="vote", provider="a", model="m", attempts=(a1,)),
        UsageRecord(agent_id="p02", task_type="vote", provider="b", model="m", attempts=(b1,)),
        UsageRecord(agent_id="p01", task_type="vote", provider="a", model="m", attempts=(a1, a2)),
    ]

    summary = _reasoning_evidence_summary(usage)

    assert [
        (item["opaque_request_id"], item["ordinal"])
        for item in summary["attempts"]
    ] == [(request_a.value, 1), (request_a.value, 2), (request_b.value, 1)]


def test_real_player_repair_usage_does_not_duplicate_reasoning_denominator():
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
    from werewolf_agent.model_gateway.router import GenerateResult, ModelRouter, UsageRecord
    from scripts.run_real_game_reports import _reasoning_evidence_summary

    class SequenceProvider:
        def __init__(self):
            self.responses = [
                "not-json",
                '{"action_type":"no_action","target_id":null,'
                '"speech":"我暂不投票，继续观察公开发言。",'
                '"reason":"当前证据不足以支持投票。","confidence":0.5}',
            ]

        @property
        def name(self):
            return "sequence"

        def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
            text = self.responses.pop(0)
            return GenerateResult(
                text=text,
                provider=self.name,
                model=config.model,
                tool_call_received=bool(tool_choice),
                usage=UsageRecord(
                    agent_id="p01", task_type="vote",
                    provider=self.name, model=config.model,
                ),
            )

    router = ModelRouter(
        model_profiles={
            "primary": {
                "provider": "sequence", "model": "m", "retry_count": 0,
                "reasoning": {"level": "high"},
            },
        },
        llm_profiles={
            "profile": {
                "default": {"provider": "sequence", "model_profile": "primary"},
            },
        },
        player_assignments={"p01": "profile"},
        providers={"sequence": SequenceProvider()},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=2)
    action, _ = agent.act(AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
        legal_targets=["p02"],
    ))

    assert action.trace is not None
    summary = _reasoning_evidence_summary(
        router.get_usage_log(),
        action_attempts=action.trace.execution_attempts,
    )
    assert summary["requested_denominator"] == 2
    assert [item["route"] for item in summary["attempts"]] == ["primary", "repair"]
    assert summary["attempts"][0]["root_cause"] == "invalid_output"
    assert summary["attempts"][0]["outcome"] == "attempt_failure"


def test_report_helpers_are_split_from_run_real_game_facade() -> None:
    from scripts import run_real_game, run_real_game_reports

    assert run_real_game.print_game_summary is run_real_game_reports.print_game_summary
    assert run_real_game.print_usage_stats is run_real_game_reports.print_usage_stats
    assert run_real_game.print_pace_report is run_real_game_reports.print_pace_report
    assert run_real_game.print_quality_audit is run_real_game_reports.print_quality_audit
    assert run_real_game.check_leakage is run_real_game_reports.check_leakage


def test_quality_audit_handles_vote_trace_without_parsed_action(capsys) -> None:
    runner = SimpleNamespace(
        state=GameState(
            game_id="g_trace_none",
            phase="day",
            events=[
                GameEvent(
                    type="action_trace_audit",
                    payload={
                        "phase": "vote",
                        "action_trace": {
                            "parsed_action": None,
                            "fallback_reason": "model_failed",
                        },
                    },
                )
            ],
        )
    )

    print_quality_audit(runner)

    out = capsys.readouterr().out
    assert "Votes without basis" in out
    assert "Fallbacks:" in out


def test_format_api_key_status_does_not_expose_key_material() -> None:
    from scripts.run_real_game import _format_api_key_status

    key = "sk-test-1234567890abcdef"

    status = _format_api_key_status(key)

    assert status == "configured"
    assert key[:8] not in status


def test_save_game_log_exports_complete_death_fields(tmp_path, monkeypatch) -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_export",
        players={"hunter": PlayerState(id="hunter", role="hunter", alive=False)},
        deaths=[
            Death(
                player_id="hunter",
                reason="exile",
                timing="day_vote",
                resolution_batch=ResolutionBatchV2("day", 3, "vote"),
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=["hunter_shot"],
            ),
        ],
    )
    runner = SimpleNamespace(game_id="g_export", state=gs, step_count=1)
    monkeypatch.setattr(run_real_game, "ROOT", tmp_path)

    path = _save_game_log(run_real_game, runner, elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["deaths"] == [
        {
            "player_id": "hunter",
            "reason": "exile",
            "timing": "day_vote",
            "resolution_batch": {"phase": "day", "number": 3, "cause": "vote"},
            "resolution_batch_parse_failed": False,
            "source_player_id": None,
            "can_leave_last_words": True,
            "triggered_skills": ["hunter_shot"],
        },
    ]


def test_save_game_log_exports_complete_safe_v2_event_metadata(tmp_path) -> None:
    from datetime import datetime, timezone

    from scripts import run_real_game
    from werewolf_agent.core.event_visibility import EventVisibility

    event = GameEvent(
        type="seer_check",
        payload={"target_id": "p02"},
        visibility=EventVisibility.SEER_PRIVATE,
        event_id="g_export:e000000",
        sequence_number=0,
        occurred_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        game_id="g_export",
        trace_id="trace-1",
        schema_version="2",
    )
    runner = SimpleNamespace(
        game_id="g_export",
        state=GameState(game_id="g_export", events=[event]),
        step_count=1,
    )

    path = _save_game_log(
        run_real_game,
        runner,
        elapsed=0.1,
        output_dir=tmp_path,
    )
    exported = json.loads(path.read_text(encoding="utf-8"))["events"][0]

    assert exported == {
        "type": "seer_check",
        "payload": {"target_id": "p02"},
        "visibility": "seer_private",
        "event_id": "g_export:e000000",
        "sequence_number": 0,
        "occurred_at": "2026-07-15T00:00:00+00:00",
        "game_id": "g_export",
        "trace_id": "trace-1",
        "schema_version": "2",
    }


def test_serialize_event_for_log_canonicalizes_nested_resolution_batches() -> None:
    from scripts import run_real_game
    from werewolf_agent.core.event_visibility import EventVisibility

    batch = ResolutionBatchV2("night", 2, "hunter_shot")
    event = GameEvent(
        type="nested_batch",
        payload={"items": [{"batch": batch}]},
        visibility=EventVisibility.MODERATOR_ONLY,
        schema_version="2",
    )

    exported = run_real_game._serialize_event_for_log(event)

    assert exported["payload"] == {
        "items": [
            {"batch": {"phase": "night", "number": 2, "cause": "hunter_shot"}}
        ]
    }
    assert exported["visibility"] == "moderator_only"
    assert event.payload["items"][0]["batch"] is batch
    json.dumps(exported)


def test_save_game_log_handles_nested_resolution_batches(tmp_path) -> None:
    from scripts import run_real_game

    batch = ResolutionBatchV2("day", 4, "rule_effect")
    runner = SimpleNamespace(
        game_id="g_nested_batch",
        state=GameState(
            game_id="g_nested_batch",
            events=[
                GameEvent(
                    type="nested_batch",
                    payload={"outer": {"batches": [batch]}},
                )
            ],
        ),
        step_count=1,
    )

    path = _save_game_log(run_real_game, runner, 0.1, output_dir=tmp_path)
    exported = json.loads(path.read_text(encoding="utf-8"))["events"][0]

    assert exported["payload"] == {
        "outer": {
            "batches": [
                {"phase": "day", "number": 4, "cause": "rule_effect"}
            ]
        }
    }
    assert runner.state.events[0].payload["outer"]["batches"][0] is batch


def test_save_game_log_drops_reflection_payload_visibility_for_v2(tmp_path) -> None:
    from datetime import datetime, timezone

    from scripts import run_real_game
    from werewolf_agent.core.event_visibility import EventVisibility

    event = GameEvent(
        type="reflection_complete",
        payload={
            "visibility": "public",
            "entries": [{
                "player_id": "p01",
                "decision_id": "reflection:g1:p01",
                "verification": {"status": "verified"},
            }],
        },
        visibility=EventVisibility.MODERATOR_ONLY,
        event_id="g1:e000000",
        sequence_number=0,
        occurred_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        game_id="g1",
        schema_version="2",
    )
    runner = SimpleNamespace(
        game_id="g1",
        state=GameState(game_id="g1", events=[event]),
        step_count=1,
    )

    path = _save_game_log(run_real_game, runner, 0.1, output_dir=tmp_path)
    exported = json.loads(path.read_text(encoding="utf-8"))["events"][0]

    assert exported["visibility"] == "moderator_only"
    assert "visibility" not in exported["payload"]


def test_save_game_log_keeps_v1_reflection_private_by_legacy_payload(tmp_path) -> None:
    from scripts import run_real_game

    event = GameEvent(
        type="reflection_complete",
        payload={"entries": []},
    )
    runner = SimpleNamespace(
        game_id="g1",
        state=GameState(game_id="g1", events=[event]),
        step_count=1,
    )

    path = _save_game_log(run_real_game, runner, 0.1, output_dir=tmp_path)
    exported = json.loads(path.read_text(encoding="utf-8"))["events"][0]

    assert exported["visibility"] is None
    assert exported["payload"]["visibility"] == "moderator_only"


def test_reports_classify_v2_private_event_from_top_level_visibility(capsys) -> None:
    from scripts.run_real_game_reports import check_leakage, print_game_summary
    from werewolf_agent.core.event_visibility import EventVisibility

    runner = SimpleNamespace(
        state=GameState(game_id="g-report", events=[GameEvent(
            type="seer_check",
            payload={"seer_id": "p01", "target_id": "p02", "alignment": "wolf"},
            visibility=EventVisibility.SEER_PRIVATE,
            schema_version="2",
        )]),
        step_count=1,
    )

    print_game_summary(runner)
    summary = capsys.readouterr().out
    check_leakage(runner)
    leakage = capsys.readouterr().out

    assert "[seer_private]" in summary
    assert "Seer check leaked" not in leakage
    assert "No public-state information leaks detected." in leakage


def test_reports_fail_closed_for_unknown_legacy_visibility(capsys) -> None:
    from scripts.run_real_game_reports import check_leakage, print_game_summary

    runner = SimpleNamespace(
        state=GameState(game_id="g-report", events=[GameEvent(
            type="seer_check",
            payload={
                "seer_id": "p01",
                "target_id": "p02",
                "alignment": "wolf",
                "visibility": "future_private",
            },
        )]),
        step_count=1,
    )

    print_game_summary(runner)
    summary = capsys.readouterr().out
    check_leakage(runner)
    leakage = capsys.readouterr().out

    assert "[moderator_only]" in summary
    assert "Seer check leaked" not in leakage
    assert "No public-state information leaks detected." in leakage


def test_real_game_parser_accepts_output_directory(tmp_path) -> None:
    from scripts import run_real_game

    args = run_real_game._build_argument_parser().parse_args([
        "--output-dir", str(tmp_path / "artifact"),
    ])

    assert args.output_dir == tmp_path / "artifact"


def test_real_game_parser_passes_explicit_game_id_into_runner_config() -> None:
    from scripts import run_real_game

    args = run_real_game._build_argument_parser().parse_args([
        "--seed", "713001",
        "--game-id", "audit-run-abc-seed-713001",
    ])

    config = run_real_game._build_runner_config(
        args,
        game_repo=None,
        memory_coordinator=None,
    )

    assert config.seed == 713001
    assert config.game_id == "audit-run-abc-seed-713001"


def test_real_game_runner_config_rejects_unsafe_cli_game_id() -> None:
    from scripts import run_real_game

    args = run_real_game._build_argument_parser().parse_args([
        "--seed", "713001",
        "--game-id", "../audit-escape",
    ])

    with pytest.raises(ValueError, match="game_id"):
        run_real_game._build_runner_config(
            args,
            game_repo=None,
            memory_coordinator=None,
        )


def test_save_game_log_uses_explicit_output_directory(tmp_path) -> None:
    from scripts import run_real_game

    runner = SimpleNamespace(
        game_id="g_isolated",
        state=GameState(game_id="g_isolated"),
        step_count=1,
    )
    output_dir = tmp_path / "artifact"

    path = _save_game_log(
        run_real_game, runner, elapsed=0.1, output_dir=output_dir
    )

    assert path == output_dir / "game_g_isolated.json"
    assert path.exists()


def test_low_quality_game_stays_under_explicit_output_directory(
    tmp_path, monkeypatch
) -> None:
    from scripts import run_real_game

    runner = SimpleNamespace(
        game_id="g_low",
        state=GameState(game_id="g_low"),
        step_count=1,
    )
    quality_score = {
        "fallback_rate": 1.0,
        "total_quality_events": 6,
    }
    output_dir = tmp_path / "artifact"

    path = _save_game_log(
        run_real_game,
        runner,
        elapsed=0.1,
        output_dir=output_dir,
        quality_score=quality_score,
    )

    assert path == output_dir / "low_quality_games" / "game_g_low.json"
    assert path.exists()


def test_save_game_log_path_uses_projection_id_when_runner_drifts(tmp_path) -> None:
    from scripts import run_real_game

    runner = SimpleNamespace(
        game_id="runner-drift",
        state=GameState(
            game_id="projection-id",
            players={"p01": PlayerState(id="p01", role="villager")},
        ),
        step_count=1,
    )
    projection = run_real_game.project_acceptance_game(runner.state, steps=1)
    quality = run_real_game.compute_game_quality_score(projection)

    path = run_real_game.save_game_log(
        runner,
        0.1,
        projection=projection,
        quality_score=quality,
        output_dir=tmp_path,
    )

    assert path.name == "game_projection-id.json"
    assert json.loads(path.read_text(encoding="utf-8"))["game_id"] == "projection-id"


def test_quality_score_counts_wolf_team_plan_fallbacks() -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_quality_wolf_plan",
        events=[
            GameEvent(
                type="action_trace_audit",
                payload={"action_trace": {"fallback_reason": "empty_response"}},
            ),
            GameEvent(type="wolf_team_plan_fallback", payload={"night_number": 1}),
            GameEvent(type="wolf_team_plan", payload={"night_number": 1}),
        ],
    )
    runner = SimpleNamespace(state=gs, step_count=3)

    quality = run_real_game.compute_game_quality_score(runner)

    assert quality["action_fallback_count"] == 1
    assert quality["wolf_team_plan_fallback_count"] == 1
    assert quality["fallback_count"] == 2
    assert quality["total_wolf_team_plans"] == 1
    assert quality["total_quality_events"] == 2
    assert quality["fallback_rate"] == 1.0


def test_quality_score_reports_wolf_plan_outcomes_and_null_rates_without_plans() -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_quality_wolf_outcomes",
        events=[
            GameEvent(
                type="wolf_team_plan",
                payload={"normalization_repairs": ["synthesize:public_story"]},
            ),
            GameEvent(
                type="wolf_team_plan_fallback",
                payload={"reason": "schema_validation_failed"},
            ),
            GameEvent(type="wolf_team_plan", payload={"consensus_method": "fallback"}),
        ],
    )
    quality = run_real_game.compute_game_quality_score(SimpleNamespace(state=gs, step_count=2))

    assert quality["wolf_team_plan_outcome_metrics_supported"] is True
    assert quality["wolf_team_plan_total_count"] == 2
    assert quality["wolf_team_plan_normalization_success_count"] == 1
    assert quality["wolf_team_plan_schema_terminal_fallback_count"] == 1
    assert quality["wolf_team_plan_strategy_terminal_fallback_count"] == 0
    assert quality["wolf_team_plan_normalization_triggered_count"] == 1
    assert quality["wolf_team_plan_normalization_success_rate"] == 1.0

    empty = run_real_game.compute_game_quality_score(SimpleNamespace(
        state=GameState(game_id="g_quality_no_wolf_plans"),
        step_count=0,
    ))
    assert empty["wolf_team_plan_outcome_metrics_supported"] is False
    assert empty["wolf_team_plan_normalization_success_rate"] is None


def test_quality_score_groups_fallbacks_by_reason_and_stage() -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_quality_fallback_reasons",
        events=[
            GameEvent(
                type="action_trace_audit",
                payload={
                    "action_trace": {
                        "fallback_reason": "fallback: 结构化输出失败，按当前可见线索选择默认目标",
                        "structured_failure_reason": "speech_quality",
                        "structured_failure_stage": "semantic",
                        "retry": {"error_code": "speech_quality"},
                    }
                },
            ),
            GameEvent(
                type="action_trace_audit",
                payload={
                    "action_trace": {
                        "fallback_reason": "fallback: 结构化输出失败，按当前可见线索选择默认目标",
                        "structured_failure_reason": "parse_error",
                        "structured_failure_stage": "protocol",
                        "retry": {"error_code": "parse_error"},
                    }
                },
            ),
            GameEvent(
                type="action_trace_audit",
                payload={
                    "action_trace": {
                        "structured_failure_reason": "empty_response",
                        "structured_failure_stage": "model_output",
                        "retry": {"error_code": "empty_response"},
                    }
                },
            ),
            GameEvent(
                type="wolf_team_plan_fallback",
                payload={
                    "night_number": 1,
                    "reason": "empty_response",
                    "stage": "model_output",
                },
            ),
            GameEvent(type="wolf_team_plan", payload={"night_number": 1}),
        ],
    )
    runner = SimpleNamespace(state=gs, step_count=5)

    quality = run_real_game.compute_game_quality_score(runner)

    assert quality["action_fallback_by_error_code"] == {
        "parse_error": 1,
        "speech_quality": 1,
    }
    assert quality["retry_error_counts"] == {
        "empty_response": 1,
        "parse_error": 1,
        "speech_quality": 1,
    }
    assert quality["wolf_team_plan_fallback_by_reason"] == {"empty_response": 1}
    assert quality["fallback_by_reason"] == {
        "empty_response": 1,
        "parse_error": 1,
        "speech_quality": 1,
    }
    assert quality["fallback_by_stage"] == {
        "model_output": 1,
        "protocol": 1,
        "semantic": 1,
    }


def test_save_game_log_exports_hybrid_fields_from_victory_event(tmp_path, monkeypatch) -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_hybrid_export",
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p12": PlayerState(id="p12", role="hybrid", alive=True),
        },
        winning_faction="werewolf",
        events=[
            GameEvent(
                type="victory",
                payload={
                    "winner": "werewolf",
                    "hybrid_master_id": "p01",
                    "hybrid_master_faction": "werewolf",
                    "hybrid_result": "win",
                },
            ),
        ],
    )
    runner = SimpleNamespace(game_id="g_hybrid_export", state=gs, step_count=1)
    monkeypatch.setattr(run_real_game, "ROOT", tmp_path)

    path = _save_game_log(run_real_game, runner, elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["hybrid_master_id"] == "p01"
    assert data["hybrid_master_faction"] == "werewolf"
    assert data["hybrid_result"] == "win"
