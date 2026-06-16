"""Improvement candidate generation tests."""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import FailureDiagnosis


class _ExplodingStore:
    def save_rag_entries(self, *_args, **_kwargs):
        raise AssertionError("candidate generation must not mutate RAG stores")

    def store_v2(self, *_args, **_kwargs):
        raise AssertionError("candidate generation must not mutate memory stores")

    def save_reflection(self, *_args, **_kwargs):
        raise AssertionError("candidate generation must not mutate persistence stores")


def _diagnosis(
    diagnosis_id: str,
    category: str,
    *,
    trace_id: str = "t1",
    module: str = "rag",
) -> FailureDiagnosis:
    return FailureDiagnosis(
        diagnosis_id=diagnosis_id,
        trace_id=trace_id,
        category=category,
        severity="high",
        primary_module=module,
        evidence_refs=[f"trace:{trace_id}"],
        explanation=f"{category} happened",
    )


def test_generate_improvement_candidates_groups_diagnoses_into_prompt_safe_candidates() -> None:
    from werewolf_agent.evaluation.candidates import generate_improvement_candidates
    from werewolf_agent.evaluation.feedback_schemas import validate_candidate_prompt_safe

    diagnoses = [
        _diagnosis("d1", "rag_harmful_transfer", module="rag"),
        _diagnosis("d2", "reflection_harmful_transfer", module="reflection"),
        _diagnosis("d3", "low_true_world_rank", module="possible_worlds"),
        _diagnosis("d4", "simulator_false_positive", module="simulator"),
    ]

    candidates = generate_improvement_candidates(diagnoses)
    by_module = {candidate.target_module: candidate for candidate in candidates}

    assert by_module["rag"].operation == "review_or_rewrite"
    assert by_module["reflection"].operation == "quarantine_or_rewrite"
    assert by_module["possible_worlds"].operation == "adjust_ranking_features"
    assert by_module["simulator"].operation == "calibrate_prediction_filter"
    assert all(validate_candidate_prompt_safe(candidate) for candidate in candidates)
    assert all(candidate.audit_evidence["diagnosis_count"] == 1 for candidate in candidates)


def test_generate_improvement_candidates_does_not_mutate_live_stores() -> None:
    from werewolf_agent.evaluation.candidates import generate_improvement_candidates

    candidates = generate_improvement_candidates(
        [_diagnosis("d1", "rag_harmful_transfer", module="rag")],
        rag_store=_ExplodingStore(),
        memory_store=_ExplodingStore(),
    )

    assert len(candidates) == 1


def test_generate_improvement_candidates_groups_multiple_diagnoses_per_module() -> None:
    from werewolf_agent.evaluation.candidates import generate_improvement_candidates

    candidates = generate_improvement_candidates([
        _diagnosis("d1", "rag_harmful_transfer", trace_id="t1", module="rag"),
        _diagnosis("d2", "rag_harmful_transfer", trace_id="t2", module="rag"),
    ])

    assert len(candidates) == 1
    assert candidates[0].target_module == "rag"
    assert candidates[0].source_diagnosis_ids == ["d1", "d2"]
    assert candidates[0].audit_evidence["diagnosis_count"] == 2
    assert candidates[0].audit_evidence["trace_ids"] == ["t1", "t2"]


def test_generate_improvement_candidates_drops_prompt_unsafe_payloads() -> None:
    from werewolf_agent.evaluation.candidates import generate_improvement_candidates

    diagnosis = _diagnosis(
        "d1",
        "hidden_info_leak",
        module="prompt_builder",
    )

    candidates = generate_improvement_candidates(
        [diagnosis],
        payload_overrides={
            "prompt_builder": {"recommended_use": "根据 p03 的真实身份调整"}
        },
    )

    assert candidates == []
