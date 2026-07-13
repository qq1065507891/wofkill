from __future__ import annotations

from werewolf_agent.evaluation.schemas import GameResult
from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder


def _result(event_log: list[dict]) -> GameResult:
    return GameResult(
        game_id="g_runtime_exposure",
        initial_seed=11,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p01": "villager", "p02": "werewolf"},
        player_factions={"p01": "good", "p02": "werewolf"},
        event_log=event_log,
    )


def _action_event(trace_id: str = "g_runtime_exposure:p01:vote:D2:N1:vote:7") -> dict:
    return {
        "type": "action_trace_audit",
        "payload": {
            "trace_id": trace_id,
            "player_id": "p01",
            "phase": "vote",
            "day_number": 2,
            "night_number": 1,
            "task_type": "vote",
            "action_index": 7,
            "action_trace": {
                "task_type": "vote",
                "final_action_type": "vote",
                "parsed_action": {
                    "action_type": "vote",
                    "target_id": "p02",
                    "reason": "runtime exposure join",
                },
            },
        },
    }


def test_trace_builder_joins_runtime_exposure_events_without_side_channel() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"
    traces = EvaluationTraceBuilder().build(
        _result([
            {
                "type": "rag_exposure_audit",
                "payload": {
                    "trace_id": trace_id,
                    "hits": [
                        {
                            "entry_id": "rag_vote_anchor",
                            "rank": 1,
                            "relevance_score": 0.74,
                            "prompt_visible": True,
                        }
                    ],
                },
            },
            {
                "type": "reflection_exposure_audit",
                "payload": {
                    "trace_id": trace_id,
                    "cards": [
                        {
                            "entry_id": "reflection_vote_anchor",
                            "rank": 2,
                            "quality_score": 0.81,
                            "prompt_visible": True,
                        }
                    ],
                },
            },
            {
                "type": "skill_exposure_audit",
                "payload": {
                    "trace_id": trace_id,
                    "analyses": [
                        {
                            "skill_name": "vote_analysis",
                            "rank": 1,
                            "prompt_visible": True,
                            "summary_hash": "sha256:abc",
                            "advice_type": "tactical",
                        }
                    ],
                },
            },
            {
                "type": "persona_exposure_audit",
                "payload": {
                    "trace_id": trace_id,
                    "snapshot": {
                        "profile_id": "logic_leader",
                        "prompt_visible": True,
                        "policy_keys": ["risk"],
                        "sanitized": True,
                    },
                },
            },
            _action_event(trace_id),
        ])
    )

    assert traces[0].trace_id == trace_id
    exposures = {(exposure.module, exposure.item_id): exposure for exposure in traces[0].module_exposures}
    assert ("rag", "rag_vote_anchor") in exposures
    assert ("reflection", "reflection_vote_anchor") in exposures
    assert ("skills", "vote_analysis") in exposures
    assert ("persona", "logic_leader") in exposures
    assert ("rag", "missing_source") not in exposures
    assert ("reflection", "missing_source") not in exposures
    assert exposures[("skills", "vote_analysis")].metadata["advice_type"] == "tactical"
    assert exposures[("persona", "logic_leader")].metadata["policy_keys"] == ["risk"]


def test_trace_builder_keeps_side_channel_exposure_compatibility() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"

    traces = EvaluationTraceBuilder().build(
        _result([_action_event(trace_id)]),
        exposure_audits=[
            {
                "type": "rag_exposure_audit",
                "trace_id": trace_id,
                "hits": [{"entry_id": "side_channel_rag", "prompt_visible": True}],
            }
        ],
    )

    assert {
        (exposure.module, exposure.item_id)
        for exposure in traces[0].module_exposures
    } >= {("rag", "side_channel_rag")}


def test_runtime_event_type_overrides_payload_type_field() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"

    traces = EvaluationTraceBuilder().build(
        _result([
            {
                "type": "rag_exposure_audit",
                "payload": {
                    "type": "not_an_audit_type",
                    "trace_id": trace_id,
                    "hits": [{"entry_id": "rag_type_conflict", "prompt_visible": True}],
                },
            },
            _action_event(trace_id),
        ])
    )

    exposures = {(exposure.module, exposure.item_id) for exposure in traces[0].module_exposures}
    assert ("rag", "rag_type_conflict") in exposures
    assert ("rag", "missing_source") not in exposures


def test_skill_exposure_metadata_keeps_only_safe_summary_fields() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"

    traces = EvaluationTraceBuilder().build(
        _result([_action_event(trace_id)]),
        exposure_audits=[
            {
                "type": "skill_exposure_audit",
                "trace_id": trace_id,
                "analyses": [
                    {
                        "skill_name": "vote_analysis",
                        "rank": 1,
                        "prompt_visible": True,
                        "summary_hash": "sha256:safe",
                        "advice_type": "tactical",
                        "raw_prompt": "private prompt",
                        "private_reason": "hidden chain",
                        "analysis": "raw advice text",
                    }
                ],
            }
        ],
    )

    exposure = next(
        exposure
        for exposure in traces[0].module_exposures
        if exposure.module == "skills"
    )
    assert exposure.metadata == {
        "summary_hash": "sha256:safe",
        "advice_type": "tactical",
    }


def test_skill_tool_call_audit_joins_trace_with_safe_call_details() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"

    traces = EvaluationTraceBuilder().build(
        _result([
            {
                "type": "skill_tool_call_audit",
                "payload": {
                    "trace_id": trace_id,
                    "calls": [
                        {
                            "call_kind": "skill",
                            "call_name": "push_vote",
                            "skill_name": "push_vote",
                            "status": "success",
                            "success": True,
                            "prompt_visible": True,
                            "result_available_to_decision": True,
                            "decision_usage": "prompt_injected",
                            "input_summary": {
                                "role": "villager",
                                "task_type": "vote",
                                "private_role": "werewolf",
                            },
                            "output_summary": {
                                "confidence": 0.82,
                                "risk_alert_count": 1,
                                "private_reasoning": "hidden",
                            },
                            "private_payload": "must not leak",
                        },
                        {
                            "call_kind": "tool",
                            "call_name": "submit_player_action",
                            "tool_name": "submit_player_action",
                            "status": "missing",
                            "success": False,
                            "required": True,
                            "received": False,
                            "fallback_triggered": True,
                            "decision_usage": "not_used_fallback",
                            "structured_failure_reason": "missing_tool_call",
                        },
                    ],
                },
            },
            _action_event(trace_id),
        ])
    )

    exposures = {
        (exposure.module, exposure.item_id): exposure
        for exposure in traces[0].module_exposures
    }
    skill_call = exposures[("skill_tool_calls", "push_vote")]
    tool_call = exposures[("skill_tool_calls", "submit_player_action")]
    assert skill_call.prompt_visible is True
    assert skill_call.score == 1.0
    assert skill_call.metadata == {
        "call_kind": "skill",
        "status": "success",
        "success": True,
        "result_available_to_decision": True,
        "decision_usage": "prompt_injected",
        "input_summary": {
            "role": "villager",
            "task_type": "vote",
        },
        "output_summary": {
            "confidence": 0.82,
            "risk_alert_count": 1,
        },
    }
    assert tool_call.score == 0.0
    assert tool_call.metadata["required"] is True
    assert tool_call.metadata["received"] is False
    assert tool_call.metadata["fallback_triggered"] is True
    assert tool_call.metadata["structured_failure_reason"] == "missing_tool_call"
    assert "private" not in str(skill_call.metadata)


def test_prompt_injection_audit_joins_trace_with_safe_metadata() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"

    traces = EvaluationTraceBuilder().build(
        _result([
            {
                "type": "prompt_injection_audit",
                "payload": {
                    "trace_id": trace_id,
                    "injections": [
                        {
                            "module_name": "public_summary",
                            "field_path": "public_summary",
                            "injection_kind": "text_section",
                            "injected": True,
                            "prompt_visible": True,
                            "visibility_scope": "public",
                            "item_count": 1,
                            "char_count": 120,
                            "content_hash": "abc123",
                            "decision_usage": "prompt_context_available",
                            "raw_content": "private text must not leak",
                        },
                        {
                            "module_name": "strategy_directive",
                            "field_path": "strategy_directive",
                            "injection_kind": "structured_section",
                            "injected": False,
                            "prompt_visible": True,
                            "visibility_scope": "viewer_visible",
                            "item_count": 0,
                            "char_count": 0,
                            "content_hash": "",
                            "decision_usage": "not_available_empty",
                        },
                    ],
                },
            },
            _action_event(trace_id),
        ])
    )

    exposures = {
        (exposure.module, exposure.item_id): exposure
        for exposure in traces[0].module_exposures
    }
    public_summary = exposures[("prompt_injections", "public_summary")]
    strategy = exposures[("prompt_injections", "strategy_directive")]
    assert public_summary.prompt_visible is True
    assert public_summary.score == 1.0
    assert public_summary.metadata == {
        "module_name": "public_summary",
        "field_path": "public_summary",
        "injection_kind": "text_section",
        "injected": True,
        "visibility_scope": "public",
        "item_count": 1,
        "char_count": 120,
        "content_hash": "abc123",
        "decision_usage": "prompt_context_available",
    }
    assert strategy.score == 0.0
    assert strategy.metadata["decision_usage"] == "not_available_empty"
    assert "private text" not in str(public_summary.metadata)


def test_persona_exposure_metadata_keeps_only_safe_policy_fields() -> None:
    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"

    traces = EvaluationTraceBuilder().build(
        _result([_action_event(trace_id)]),
        exposure_audits=[
            {
                "type": "persona_exposure_audit",
                "trace_id": trace_id,
                "snapshot": {
                    "profile_id": "logic_leader",
                    "prompt_visible": True,
                    "policy_keys": ["risk"],
                    "sanitized": True,
                    "effective_params": {"deception_skill": 0.9},
                    "private_note": "hidden",
                },
            }
        ],
    )

    exposure = next(
        exposure
        for exposure in traces[0].module_exposures
        if exposure.module == "persona"
    )
    assert exposure.metadata == {
        "policy_keys": ["risk"],
        "sanitized": True,
    }


def test_persona_provider_proof_flows_through_trace_builder_and_feedback_report() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report

    trace_id = "g_runtime_exposure:p01:vote:D2:N1:vote:7"
    events = [
        {
            "type": "persona_exposure_audit",
            "payload": {
                "trace_id": trace_id,
                "snapshot": {"profile_id": "logic_leader", "sanitized": True},
            },
        },
        {
            "type": "persona_prompt_injection_audit",
            "payload": {
                "trace_id": trace_id,
                "proof": {
                    "final_system_location": "system",
                    "final_system_message_index": None,
                    "message_char_count": 42,
                    "run_scoped_fingerprint": "run_hmac_opaque",
                    "confirmed_injection": True,
                    "attempt_kind": "provider_fallback",
                    "attempt_ordinal": 2,
                    "provider": "anthropic",
                    "model": "fallback-model",
                    "sanitized": True,
                },
            },
        },
        # 同一 decision 的重复 provider attempt 不得重复计入 confirmed action。
        {
            "type": "persona_prompt_injection_audit",
            "payload": {
                "trace_id": trace_id,
                "proof": {
                    "final_system_location": "messages",
                    "final_system_message_index": 0,
                    "message_char_count": 42,
                    "run_scoped_fingerprint": "run_hmac_other",
                    "confirmed_injection": True,
                    "attempt_kind": "provider_fallback",
                    "attempt_ordinal": 3,
                    "provider": "openai",
                    "model": "fallback-model-2",
                    "sanitized": True,
                },
            },
        },
        _action_event(trace_id),
    ]

    traces = EvaluationTraceBuilder().build(_result(events))
    report = build_feedback_report(report_id="r1", batch_id="b1", traces=traces)
    payload = report.to_json_dict()

    proof_rows = [
        row for row in payload["monitoring_exposures"]
        if row["module"] == "persona_prompt_confirmation"
    ]
    assert len(proof_rows) == 2
    assert payload["persona_prompt_confirmation"] == {
        "supported": True,
        "configured_action_count": 1,
        "confirmed_action_count": 1,
        "confirmation_rate": 1.0,
    }
