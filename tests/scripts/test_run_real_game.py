# -*- coding: utf-8 -*-
"""
验证真实游戏脚本的报告辅助函数与结构化质量指标。

作者: Project contributors
修改日期: 2026-07-13
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.run_real_game import _safe_event_payload, print_quality_audit
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState


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
        "player_count": 1,
        "entries": [{
            "player_id": "p01",
            "role": "seer",
            "decision_id": "reflection:g1:p01",
            "reflection": "RAW_PROVIDER_DRAFT",
            "provider_response": {"thinking": "SECRET"},
            "verification": {
                "status": "verified", "decision_id": "reflection:g1:p01",
                "verified_fact_count": 1,
                "verified_lessons": [{"lesson_id": "l1", "abstraction": "先复核公开票型"}],
                "rejected_fact_count": 0, "rejected_lesson_count": 0,
            },
        }],
    }

    safe = _safe_event_payload("reflection_complete", payload)
    serialized = json.dumps(safe, ensure_ascii=False)

    assert "RAW_PROVIDER_DRAFT" not in serialized
    assert "SECRET" not in serialized
    assert safe["entries"][0]["verification"]["verified_fact_count"] == 1
    assert safe["entries"][0]["decision_id"] == "reflection:g1:p01"
    assert safe["entries"][0]["verification"]["decision_id"] == "reflection:g1:p01"


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
                resolution_batch="day_3_vote",
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=["hunter_shot"],
            ),
        ],
    )
    runner = SimpleNamespace(game_id="g_export", state=gs, step_count=1)
    monkeypatch.setattr(run_real_game, "ROOT", tmp_path)

    path = run_real_game.save_game_log(runner, elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["deaths"] == [
        {
            "player_id": "hunter",
            "reason": "exile",
            "timing": "day_vote",
            "resolution_batch": "day_3_vote",
            "source_player_id": None,
            "can_leave_last_words": True,
            "triggered_skills": ["hunter_shot"],
        },
    ]


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
    assert quality["wolf_team_plan_normalization_success_rate"] == 0.5

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

    path = run_real_game.save_game_log(runner, elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["hybrid_master_id"] == "p01"
    assert data["hybrid_master_faction"] == "werewolf"
    assert data["hybrid_result"] == "win"
