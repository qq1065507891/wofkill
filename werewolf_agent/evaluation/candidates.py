"""Generate reviewable improvement candidates from feedback diagnoses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from werewolf_agent.evaluation.feedback_schemas import (
    FailureDiagnosis,
    ImprovementCandidate,
    validate_candidate_prompt_safe,
)


def generate_improvement_candidates(
    diagnoses: list[FailureDiagnosis],
    *,
    rag_store: Any = None,
    memory_store: Any = None,
    payload_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[ImprovementCandidate]:
    """Generate prompt-safe candidate objects without mutating live stores."""
    del rag_store, memory_store
    grouped: dict[str, list[FailureDiagnosis]] = defaultdict(list)
    for diagnosis in diagnoses:
        target_module = _target_module(diagnosis)
        if not target_module:
            continue
        grouped[target_module].append(diagnosis)

    candidates: list[ImprovementCandidate] = []
    for target_module, group in sorted(grouped.items()):
        categories = sorted({diagnosis.category for diagnosis in group})
        payload = dict(
            (payload_overrides or {}).get(target_module)
            or _default_payload(target_module, categories)
        )
        candidate = ImprovementCandidate(
            candidate_id=_candidate_id(target_module, categories),
            source_diagnosis_ids=[diagnosis.diagnosis_id for diagnosis in group],
            target_module=target_module,
            operation=_operation(target_module, categories),
            priority=_priority(group),
            prompt_safe_payload=payload,
            audit_evidence={
                "diagnosis_count": len(group),
                "categories": categories,
                "trace_ids": sorted({diagnosis.trace_id for diagnosis in group}),
            },
            moderator_notes=(
                f"Review {target_module} because feedback diagnostics found: "
                + ", ".join(categories)
            ),
            safety_notes=[
                "Candidate is review-only and must not be auto-applied to live stores.",
            ],
        )
        if validate_candidate_prompt_safe(candidate):
            candidates.append(candidate)
    return candidates


def _target_module(diagnosis: FailureDiagnosis) -> str:
    if diagnosis.category == "hidden_info_leak":
        return diagnosis.primary_module or "prompt_builder"
    if diagnosis.category.startswith("rag_"):
        return "rag"
    if diagnosis.category.startswith("reflection_"):
        return "reflection"
    if diagnosis.category == "low_true_world_rank":
        return "possible_worlds"
    if diagnosis.category.startswith("simulator_"):
        return "simulator"
    return diagnosis.primary_module


def _operation(target_module: str, categories: list[str]) -> str:
    if target_module == "rag":
        return "review_or_rewrite"
    if target_module == "reflection":
        return "quarantine_or_rewrite"
    if target_module == "possible_worlds":
        return "adjust_ranking_features"
    if target_module == "simulator":
        return "calibrate_prediction_filter"
    if "hidden_info_leak" in categories:
        return "harden_prompt_boundary"
    if "illegal_action" in categories:
        return "tighten_action_validation"
    if "wrong_target" in categories:
        return "improve_target_selection"
    return "review"


def _default_payload(target_module: str, categories: list[str]) -> dict[str, Any]:
    return {
        "recommended_use": (
            f"Review {target_module} behavior for recurring feedback signals."
        ),
        "live_prompt_instruction": (
            "Do not apply automatically; route to moderator review before "
            "changing live prompts, RAG entries, or reflection cards."
        ),
    }


def _priority(group: list[FailureDiagnosis]) -> str:
    if any(diagnosis.severity == "high" for diagnosis in group):
        return "high"
    if len(group) >= 3:
        return "high"
    return "medium"


def _candidate_id(target_module: str, categories: list[str]) -> str:
    return "candidate:" + _clean(target_module) + ":" + "_".join(
        _clean(category) for category in categories
    )


def _clean(value: str) -> str:
    return str(value).replace(":", "_").replace(" ", "_")
