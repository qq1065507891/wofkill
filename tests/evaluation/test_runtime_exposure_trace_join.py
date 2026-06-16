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
