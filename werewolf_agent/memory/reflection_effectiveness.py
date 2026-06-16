"""Offline effectiveness evaluation for reflection memory exposures."""

from __future__ import annotations

from dataclasses import dataclass, field

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)


@dataclass(frozen=True)
class ReflectionEntryEffectiveness:
    entry_id: str
    exposure_count: int = 0
    injected_count: int = 0
    cited_count: int = 0
    aligned_count: int = 0
    stale_count: int = 0
    harmful_count: int = 0

    @property
    def citation_rate(self) -> float:
        return _rate(self.cited_count, self.exposure_count)

    @property
    def alignment_rate(self) -> float:
        return _rate(self.aligned_count, self.exposure_count)

    @property
    def stale_rate(self) -> float:
        return _rate(self.stale_count, self.exposure_count)

    @property
    def harmful_rate(self) -> float:
        return _rate(self.harmful_count, self.exposure_count)


@dataclass(frozen=True)
class ReflectionCandidateHint:
    entry_id: str
    reason: str
    suggested_operation: str
    prompt_safe_note: str
    exposure_count: int = 0
    harmful_count: int = 0


@dataclass(frozen=True)
class ReflectionEffectivenessReport:
    total_reflection_exposures: int = 0
    supported_reflection_exposures: int = 0
    unsupported_reflection_exposures: int = 0
    injected_count: int = 0
    cited_count: int = 0
    aligned_count: int = 0
    stale_count: int = 0
    harmful_count: int = 0
    by_entry: dict[str, ReflectionEntryEffectiveness] = field(default_factory=dict)
    candidate_hints: list[ReflectionCandidateHint] = field(default_factory=list)

    @property
    def injection_rate(self) -> float:
        return _rate(self.injected_count, self.supported_reflection_exposures)

    @property
    def citation_rate(self) -> float:
        return _rate(self.cited_count, self.supported_reflection_exposures)

    @property
    def alignment_rate(self) -> float:
        return _rate(self.aligned_count, self.supported_reflection_exposures)

    @property
    def stale_rate(self) -> float:
        return _rate(self.stale_count, self.supported_reflection_exposures)

    @property
    def harmful_rate(self) -> float:
        return _rate(self.harmful_count, self.supported_reflection_exposures)


def evaluate_reflection_effectiveness(
    traces: list[EvaluationTrace],
    *,
    no_effect_threshold: int = 3,
    harmful_threshold: int = 2,
) -> ReflectionEffectivenessReport:
    """Compute reflection-specific usefulness signals from evaluation traces."""
    total = 0
    supported = 0
    unsupported = 0
    injected = 0
    cited = 0
    aligned = 0
    stale = 0
    harmful = 0
    mutable_by_entry: dict[str, dict[str, int]] = {}

    for trace in traces:
        for exposure in trace.module_exposures:
            if exposure.module != "reflection":
                continue
            total += 1
            if exposure.support == MetricSupport.UNSUPPORTED:
                unsupported += 1
                continue
            supported += 1
            if exposure.prompt_visible:
                injected += 1
            if exposure.cited_by_decision:
                cited += 1
            if exposure.aligned_with_decision:
                aligned += 1
            if _is_stale(exposure):
                stale += 1
            if _is_harmful(exposure):
                harmful += 1

            entry_stats = mutable_by_entry.setdefault(
                exposure.item_id,
                {
                    "exposure_count": 0,
                    "injected_count": 0,
                    "cited_count": 0,
                    "aligned_count": 0,
                    "stale_count": 0,
                    "harmful_count": 0,
                },
            )
            entry_stats["exposure_count"] += 1
            if exposure.prompt_visible:
                entry_stats["injected_count"] += 1
            if exposure.cited_by_decision:
                entry_stats["cited_count"] += 1
            if exposure.aligned_with_decision:
                entry_stats["aligned_count"] += 1
            if _is_stale(exposure):
                entry_stats["stale_count"] += 1
            if _is_harmful(exposure):
                entry_stats["harmful_count"] += 1

    by_entry = {
        entry_id: ReflectionEntryEffectiveness(entry_id=entry_id, **stats)
        for entry_id, stats in sorted(mutable_by_entry.items())
        if entry_id
    }
    return ReflectionEffectivenessReport(
        total_reflection_exposures=total,
        supported_reflection_exposures=supported,
        unsupported_reflection_exposures=unsupported,
        injected_count=injected,
        cited_count=cited,
        aligned_count=aligned,
        stale_count=stale,
        harmful_count=harmful,
        by_entry=by_entry,
        candidate_hints=_candidate_hints(
            by_entry,
            no_effect_threshold=no_effect_threshold,
            harmful_threshold=harmful_threshold,
        ),
    )


def _candidate_hints(
    by_entry: dict[str, ReflectionEntryEffectiveness],
    *,
    no_effect_threshold: int,
    harmful_threshold: int,
) -> list[ReflectionCandidateHint]:
    hints: list[ReflectionCandidateHint] = []
    for entry_id, stats in by_entry.items():
        if stats.harmful_count >= harmful_threshold:
            hints.append(ReflectionCandidateHint(
                entry_id=entry_id,
                reason="harmful_transfer",
                suggested_operation="quarantine_or_rewrite",
                prompt_safe_note=(
                    "Reflection card repeatedly correlates with harmful transfer; "
                    "review or rewrite before future live injection."
                ),
                exposure_count=stats.exposure_count,
                harmful_count=stats.harmful_count,
            ))
            continue
        if (
            stats.exposure_count >= no_effect_threshold
            and stats.cited_count == 0
            and stats.aligned_count == 0
        ):
            hints.append(ReflectionCandidateHint(
                entry_id=entry_id,
                reason="repeated_no_effect",
                suggested_operation="downgrade_or_rewrite",
                prompt_safe_note=(
                    "Reflection card was injected repeatedly without citation "
                    "or alignment; consider lowering priority or rewriting it."
                ),
                exposure_count=stats.exposure_count,
                harmful_count=stats.harmful_count,
            ))
    return hints


def _is_stale(exposure: ModuleExposure) -> bool:
    return exposure.metadata.get("stale") is True


def _is_harmful(exposure: ModuleExposure) -> bool:
    return (
        exposure.metadata.get("harmful_transfer") is True
        or exposure.metadata.get("harmful") is True
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
