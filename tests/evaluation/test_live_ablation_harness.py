"""Live-context ablation harness tests."""

from __future__ import annotations

from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType


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
