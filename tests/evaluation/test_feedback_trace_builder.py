"""Feedback-loop trace builder tests.

These tests cover the first implementation slice of the evaluation feedback
loop: prompt-safe schemas, stable trace identity, module exposure joins, and
post-game outcome labels.
"""

from __future__ import annotations

from werewolf_agent.evaluation.schemas import GameResult


def _result(event_log: list[dict]) -> GameResult:
    return GameResult(
        game_id="g_feedback",
        initial_seed=7,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={
            "p01": "seer",
            "p02": "werewolf",
            "p03": "villager",
        },
        player_factions={
            "p01": "good",
            "p02": "werewolf",
            "p03": "good",
        },
        event_log=event_log,
    )


def _action_trace_event(*, target_id: str = "p02") -> dict:
    return {
        "type": "action_trace_audit",
        "payload": {
            "player_id": "p01",
            "phase": "vote",
            "day_number": 2,
            "night_number": 1,
            "action_trace": {
                "task_type": "vote",
                "legal_actions": ["vote", "no_action"],
                "legal_targets": ["p02", "p03"],
                "final_action_type": "vote",
                "parsed_action": {
                    "action_type": "vote",
                    "target_id": target_id,
                    "reason": "p02 pressure is highest",
                    "decision_plan": {
                        "action_type": "vote",
                        "target_id": target_id,
                    },
                },
                "world_model_audit": {
                    "possible_worlds": {
                        "type": "possible_worlds",
                        "top_worlds": [
                            {
                                "label": "World 1",
                                "probability": 0.7,
                                "key_assignments": {"p02": "werewolf"},
                            }
                        ],
                    },
                    "simulation_predictions": {
                        "type": "simulation",
                        "predictions": [
                            {
                                "event": "next_day_vote_pressure",
                                "probability": 0.8,
                                "affected_players": ["p02"],
                            }
                        ],
                    },
                },
            },
        },
    }


def test_candidate_prompt_payload_rejects_hidden_truth() -> None:
    from werewolf_agent.evaluation.feedback_schemas import (
        ImprovementCandidate,
        validate_candidate_prompt_safe,
    )

    candidate = ImprovementCandidate(
        candidate_id="c1",
        source_diagnosis_ids=["d1"],
        target_module="rag",
        operation="create",
        priority="high",
        prompt_safe_payload={"recommended_use": "根据 p03 的真实身份调整"},
        audit_evidence={"target_role": "werewolf"},
    )

    assert validate_candidate_prompt_safe(candidate) is False


def test_trace_builder_derives_stable_trace_id_and_decision_snapshot() -> None:
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    traces = EvaluationTraceBuilder().build(_result([_action_trace_event()]))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "g_feedback:p01:vote:D2:N1:vote:0"
    assert trace.player_id == "p01"
    assert trace.task_type == "vote"
    assert trace.legal_actions == ["vote", "no_action"]
    assert trace.decision is not None
    assert trace.decision.action_type == "vote"
    assert trace.decision.target_id == "p02"
    assert trace.outcome is not None
    assert trace.outcome.target_role == "werewolf"
    assert trace.outcome.target_faction == "werewolf"


def test_trace_builder_joins_rag_and_reflection_exposure_audits() -> None:
    from werewolf_agent.evaluation.feedback_schemas import MetricSupport
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    trace_id = "g_feedback:p01:vote:D2:N1:vote:0"
    traces = EvaluationTraceBuilder().build(
        _result([_action_trace_event()]),
        exposure_audits=[
            {
                "type": "rag_exposure_audit",
                "trace_id": trace_id,
                "hits": [
                    {
                        "entry_id": "seed_vote_pressure",
                        "rank": 1,
                        "relevance_score": 0.82,
                        "prompt_visible": True,
                    }
                ],
            },
            {
                "type": "reflection_exposure_audit",
                "trace_id": trace_id,
                "cards": [
                    {
                        "entry_id": "reflection_g0_p01",
                        "rank": 1,
                        "quality_score": 0.91,
                        "prompt_visible": True,
                    }
                ],
            },
        ],
    )

    exposures = {(e.module, e.item_id): e for e in traces[0].module_exposures}
    assert exposures[("rag", "seed_vote_pressure")].score == 0.82
    assert exposures[("rag", "seed_vote_pressure")].support == MetricSupport.SUPPORTED
    assert exposures[("reflection", "reflection_g0_p01")].score == 0.91
    assert exposures[("reflection", "reflection_g0_p01")].prompt_visible is True


def test_missing_exposure_sources_are_marked_unsupported_not_zero() -> None:
    from werewolf_agent.evaluation.feedback_schemas import MetricSupport
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    trace = EvaluationTraceBuilder().build(_result([_action_trace_event()]))[0]

    unsupported = {
        exposure.module: exposure
        for exposure in trace.module_exposures
        if exposure.support == MetricSupport.UNSUPPORTED
    }
    assert unsupported["rag"].item_id == "missing_source"
    assert unsupported["reflection"].item_id == "missing_source"


def test_world_model_and_simulator_exposures_from_action_trace() -> None:
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    trace = EvaluationTraceBuilder().build(_result([_action_trace_event()]))[0]

    exposures = {(e.module, e.item_id): e for e in trace.module_exposures}
    assert exposures[("possible_worlds", "World 1")].score == 0.7
    assert exposures[("possible_worlds", "World 1")].metadata["key_assignments"] == {
        "p02": "werewolf",
    }
    assert exposures[("simulator", "next_day_vote_pressure")].score == 0.8
    assert exposures[("simulator", "next_day_vote_pressure")].metadata["affected_players"] == [
        "p02",
    ]


def _build_result_with_action_trace(action_trace: dict, *, player_id: str) -> GameResult:
    """Minimal GameResult carrying a single action_trace_audit event."""
    return GameResult(
        game_id="g_outcome",
        initial_seed=11,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={},
        player_factions={},
        event_log=[
            {
                "type": "action_trace_audit",
                "payload": {
                    "player_id": player_id,
                    "phase": "speech",
                    "action_trace": action_trace,
                },
            }
        ],
    )


def test_outcome_flags_illegal_action_type():
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    action_trace = {
        "final_action_type": "vote",
        "legal_actions": ["speech"],  # vote not in legal_actions -> illegal
        "legal_targets": ["p03"],
        "parsed_action": {"target_id": "p03"},
    }
    result = _build_result_with_action_trace(action_trace, player_id="p01")
    traces = EvaluationTraceBuilder().build(result)
    assert traces, "expected at least one trace"
    assert traces[0].outcome.legal is False


def test_outcome_flags_dialogue_leak():
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    action_trace = {
        "final_action_type": "speech",
        "parsed_action": {
            "speech": "我怀疑狼队友会刀我",  # contains the 狼队友 leak marker
            "dialogue_plan": {
                "public_intent": "我怀疑狼队友会刀我",
                "conceal": ["夜刀目标"],
            },
        },
    }
    result = _build_result_with_action_trace(action_trace, player_id="p01")
    traces = EvaluationTraceBuilder().build(result)
    assert traces[0].outcome.leaked_hidden_info is True


def test_outcome_legal_unknown_without_action_type():
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    action_trace = {"parsed_action": {}}  # no action_type
    result = _build_result_with_action_trace(action_trace, player_id="p01")
    traces = EvaluationTraceBuilder().build(result)
    assert traces[0].outcome.legal is None
    assert traces[0].outcome.leaked_hidden_info is False

