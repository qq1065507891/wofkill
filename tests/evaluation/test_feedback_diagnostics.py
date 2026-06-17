"""Feedback failure diagnosis tests."""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import (
    DecisionOutcome,
    DecisionSnapshot,
    EvaluationTrace,
    ModuleExposure,
)
from werewolf_agent.evaluation.world_model_eval import WorldRankSample


def _trace(
    trace_id: str = "t1",
    *,
    decision: DecisionSnapshot | None = None,
    outcome: DecisionOutcome | None = None,
    exposures: list[ModuleExposure] | None = None,
) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id=trace_id,
        game_id="g_diag",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        legal_actions=["vote"],
        legal_targets=["p02"],
        decision=decision,
        outcome=outcome,
        module_exposures=list(exposures or []),
    )


def by_category(diagnoses):
    return {diagnosis.category: diagnosis for diagnosis in diagnoses}


def test_diagnose_trace_failures_classifies_core_decision_failures() -> None:
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures

    trace = _trace(
        decision=DecisionSnapshot(action_type="use_poison", target_id="p03"),
        outcome=DecisionOutcome(
            legal=False,
            target_faction="good",
            leaked_hidden_info=True,
        ),
    )

    diagnoses = diagnose_trace_failures([trace])
    categories = {diagnosis.category for diagnosis in diagnoses}

    assert "illegal_action" in categories
    assert "hidden_info_leak" in categories
    assert "wrong_target" in categories
    assert "harmful target" in by_category(diagnoses)["wrong_target"].explanation


def test_diagnose_trace_failures_classifies_module_transfer_and_simulator_failures() -> None:
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures

    trace = _trace(
        exposures=[
            ModuleExposure(
                module="rag",
                item_id="rag_bad",
                metadata={"harmful_transfer": True},
            ),
            ModuleExposure(
                module="reflection",
                item_id="reflection_bad",
                metadata={"harmful": True},
            ),
            ModuleExposure(
                module="simulator",
                item_id="bad_prediction",
                metadata={"false_positive": True},
            ),
        ]
    )

    diagnoses = diagnose_trace_failures([trace])
    by_category = {diagnosis.category: diagnosis for diagnosis in diagnoses}

    assert by_category["rag_harmful_transfer"].primary_module == "rag"
    assert by_category["reflection_harmful_transfer"].primary_module == "reflection"
    assert by_category["simulator_false_positive"].primary_module == "simulator"


def test_diagnose_trace_failures_classifies_low_true_world_rank_samples() -> None:
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures

    trace = _trace(trace_id="trace_low_rank")
    sample = WorldRankSample(
        trace_id="trace_low_rank",
        support="supported",
        true_world_rank=4,
        top_world_score=0.82,
    )

    diagnoses = diagnose_trace_failures(
        [trace],
        world_rank_samples=[sample],
        low_rank_threshold=3,
    )

    assert len(diagnoses) == 1
    assert diagnoses[0].category == "low_true_world_rank"
    assert diagnoses[0].primary_module == "possible_worlds"
    assert "rank=4" in diagnoses[0].explanation


def test_diagnose_trace_failures_ignores_unsupported_world_rank_samples() -> None:
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures

    trace = _trace(trace_id="trace_unsupported")
    sample = WorldRankSample(
        trace_id="trace_unsupported",
        support="unsupported",
        unsupported_reason="no_comparable_assignments",
    )

    assert diagnose_trace_failures([trace], world_rank_samples=[sample]) == []


def test_illegal_outcome_produces_illegal_action_diagnosis():
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures
    from werewolf_agent.evaluation.feedback_schemas import (
        DecisionOutcome,
        DecisionSnapshot,
        EvaluationTrace,
    )

    trace = EvaluationTrace(
        trace_id="t1",
        game_id="g1",
        player_id="p01",
        role="villager",
        faction="good",
        phase="day_vote",
        decision=DecisionSnapshot(action_type="vote", target_id="p02"),
        outcome=DecisionOutcome(legal=False, target_faction="good"),
    )
    diagnoses = diagnose_trace_failures([trace])
    categories = {d.category for d in diagnoses}
    assert "illegal_action" in categories


def test_leaked_outcome_produces_hidden_info_leak_diagnosis():
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures
    from werewolf_agent.evaluation.feedback_schemas import (
        DecisionOutcome,
        DecisionSnapshot,
        EvaluationTrace,
    )

    trace = EvaluationTrace(
        trace_id="t2",
        game_id="g1",
        player_id="p01",
        role="villager",
        faction="good",
        phase="speech",
        decision=DecisionSnapshot(action_type="speech"),
        outcome=DecisionOutcome(leaked_hidden_info=True),
    )
    diagnoses = diagnose_trace_failures([trace])
    categories = {d.category for d in diagnoses}
    assert "hidden_info_leak" in categories
