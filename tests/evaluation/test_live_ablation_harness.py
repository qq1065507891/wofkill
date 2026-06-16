"""Live-context ablation harness tests."""

from __future__ import annotations

import pytest

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    TaskType,
)
from werewolf_agent.evaluation.feedback_schemas import DecisionSnapshot


def _context() -> AgentContext:
    return AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day_vote",
        day_number=2,
        night_number=1,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p03"],
        rag_hints=[{"title": "seer claim", "summary": "p02 claimed seer"}],
        reflection_memory_hints=[{"lesson": "verify flipped roles"}],
        error_pattern_hint={"top_errors": ["overtrust_claim"]},
        possible_worlds={"worlds": [{"prob": 0.6, "roles": {"p02": "seer"}}]},
        simulation_predictions={"predictions": [{"event": "p02 attacked"}]},
        strategy_directive={
            "skill_tactical_advice": {"advice": ["pressure counter-claim"]},
            "nested": {"weights": [1, 2]},
        },
        persona_snapshot={"aggressive": 0.7, "risk": 0.4},
        skill_analyses={"vote": "compare seer lines"},
        skill_analysis_hints={"vote": "target contradictions"},
    )


def _decision(
    action_type: str,
    target_id: str | None,
    confidence: float,
) -> DecisionSnapshot:
    return DecisionSnapshot(
        action_type=action_type,
        target_id=target_id,
        confidence=confidence,
        raw={"source": "test"},
    )


def test_ablation_toggle_set_normalizes_duplicate_and_blank_modules() -> None:
    from werewolf_agent.evaluation.ablation import AblationToggleSet

    toggles = AblationToggleSet([" rag ", "", "rag", "skills"])

    assert toggles.removed_modules == ["rag", "skills"]


def test_apply_ablation_toggles_removes_context_modules_without_mutating_original() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        apply_ablation_toggles,
    )

    context = _context()

    ablated = apply_ablation_toggles(
        context,
        AblationToggleSet(
            removed_modules=[
                "rag",
                "reflection",
                "possible_worlds",
                "simulator",
                "skills",
                "persona",
            ]
        ),
    )

    assert context.rag_hints
    assert context.reflection_memory_hints
    assert context.error_pattern_hint
    assert context.possible_worlds
    assert context.simulation_predictions
    assert context.persona_snapshot
    assert context.skill_analyses
    assert context.skill_analysis_hints
    assert "skill_tactical_advice" in context.strategy_directive

    assert ablated.rag_hints == []
    assert ablated.reflection_memory_hints == []
    assert ablated.error_pattern_hint == {}
    assert ablated.possible_worlds == {}
    assert ablated.simulation_predictions == {}
    assert ablated.persona_snapshot == {}
    assert ablated.skill_analyses == {}
    assert ablated.skill_analysis_hints == {}
    assert "skill_tactical_advice" not in ablated.strategy_directive


def test_apply_ablation_toggles_deep_copies_strategy_directive() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        apply_ablation_toggles,
    )

    context = _context()

    ablated = apply_ablation_toggles(context, AblationToggleSet(["skills"]))

    assert ablated.strategy_directive["nested"] == {"weights": [1, 2]}
    assert ablated.strategy_directive["nested"] is not context.strategy_directive["nested"]
    ablated.strategy_directive["nested"]["weights"].append(3)
    assert context.strategy_directive["nested"]["weights"] == [1, 2]


def test_live_context_ablation_harness_reports_paired_decision_deltas() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    calls = []

    def runner(context: AgentContext) -> DecisionSnapshot:
        calls.append(context)
        if context.rag_hints:
            return _decision("vote", "p02", 0.8)
        return _decision("vote", "p03", 0.5)

    report = LiveContextAblationHarness(runner=runner).run(
        [_context()],
        toggles=AblationToggleSet(["rag"]),
    )

    assert report.mode == "live_context"
    assert report.removed_modules == ["rag"]
    assert report.pair_count == 1
    assert report.failed_pair_count == 0
    assert report.action_changed_count == 0
    assert report.target_changed_count == 1
    assert report.avg_confidence_delta == pytest.approx(0.3)
    assert report.unsupported_metrics["live_win_rate_delta"] == (
        "live_context_mode_decision_only"
    )
    assert report.pairs[0].baseline.target_id == "p02"
    assert report.pairs[0].ablated.target_id == "p03"
    assert calls[0].rag_hints
    assert calls[1].rag_hints == []


def test_runner_exposes_live_context_ablation_entrypoint() -> None:
    from werewolf_agent.evaluation.runner import run_live_context_ablation

    report = run_live_context_ablation(
        [_context()],
        removed_modules=["persona"],
        runner=lambda context: _decision("speech", None, 0.7),
    )

    assert report.mode == "live_context"
    assert report.removed_modules == ["persona"]


def test_live_context_ablation_reports_unknown_toggles_as_unsupported() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    report = LiveContextAblationHarness(
        runner=lambda context: _decision("vote", "p02", 0.8)
    ).run([_context()], toggles=AblationToggleSet(["rag", "unknown_module"]))

    assert report.removed_modules == ["rag", "unknown_module"]
    assert report.unsupported_metrics["unknown_module"] == (
        "unsupported_toggle:unknown_module"
    )


def test_live_context_ablation_records_failed_pairs_without_delta_impact() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    def runner(context: AgentContext) -> DecisionSnapshot:
        if context.rag_hints:
            raise RuntimeError("baseline failed")
        return _decision("vote", "p03", 0.5)

    report = LiveContextAblationHarness(runner=runner).run(
        [_context()],
        toggles=AblationToggleSet(["rag"]),
    )

    assert report.pair_count == 1
    assert report.failed_pair_count == 1
    assert report.action_changed_count == 0
    assert report.target_changed_count == 0
    assert report.avg_confidence_delta == 0.0
    assert report.pairs[0].baseline.error == "baseline failed"
    assert report.pairs[0].baseline.action_type == ""
    assert report.pairs[0].baseline.raw == {}


def test_live_context_ablation_records_empty_message_exceptions_as_failures() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    def runner(context: AgentContext) -> DecisionSnapshot:
        raise TimeoutError()

    report = LiveContextAblationHarness(runner=runner).run(
        [_context()],
        toggles=AblationToggleSet(["rag"]),
    )

    assert report.pair_count == 1
    assert report.failed_pair_count == 1
    assert report.action_changed_count == 0
    assert report.target_changed_count == 0
    assert report.avg_confidence_delta == 0.0
    assert report.pairs[0].baseline.error == "TimeoutError"
    assert report.pairs[0].ablated.error == "TimeoutError"


def test_live_context_ablation_converts_supported_runner_outputs() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    outputs = iter(
        [
            PlayerAction(
                action_type=ActionType.VOTE,
                target_id="p02",
                suspect_reason="contradiction",
                not_voting_reason="p03 less suspicious",
                private_reason="p02 attacked flipped villager",
                confidence=0.9,
            ),
            FallbackAction(action_type=ActionType.NO_ACTION, target_id=None),
            {"action_type": "speech", "target_id": None, "confidence": 0.4},
            _decision("vote", "p03", 0.6),
        ]
    )

    report = LiveContextAblationHarness(runner=lambda context: next(outputs)).run(
        [_context(), _context()],
        toggles=AblationToggleSet(["rag"]),
    )

    first, second = report.pairs
    assert first.baseline.action_type == "vote"
    assert first.baseline.target_id == "p02"
    assert first.baseline.confidence == 0.9
    assert first.ablated.action_type == "no_action"
    assert first.ablated.confidence == 0.0
    assert second.baseline.action_type == "speech"
    assert second.baseline.confidence == 0.4
    assert second.ablated.action_type == "vote"
    assert second.ablated.target_id == "p03"


def test_live_context_ablation_converts_attribute_test_doubles() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    class ActionDouble:
        action_type = "vote"
        target_id = "p02"

    report = LiveContextAblationHarness(runner=lambda context: ActionDouble()).run(
        [_context()],
        toggles=AblationToggleSet(["rag"]),
    )

    assert report.pairs[0].baseline.action_type == "vote"
    assert report.pairs[0].baseline.target_id == "p02"
    assert report.pairs[0].baseline.confidence == 0.0


def test_live_context_ablation_context_id_uses_stable_context_fields() -> None:
    from werewolf_agent.evaluation.ablation import (
        AblationToggleSet,
        LiveContextAblationHarness,
    )

    report = LiveContextAblationHarness(
        runner=lambda context: _decision("vote", "p02", 0.8)
    ).run([_context()], toggles=AblationToggleSet(["rag"]))

    assert report.pairs[0].context_id == "0:p01:vote:day_vote:D2:N1"
