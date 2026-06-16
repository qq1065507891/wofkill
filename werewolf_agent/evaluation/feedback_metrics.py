"""Module attribution metrics for feedback-loop evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)


@dataclass(frozen=True)
class ModuleAttributionSummary:
    module: str
    exposure_count: int = 0
    supported_count: int = 0
    unsupported_count: int = 0
    prompt_visible_count: int = 0
    cited_count: int = 0
    aligned_count: int = 0
    harmful_count: int = 0

    @property
    def prompt_visible_rate(self) -> float:
        return _rate(self.prompt_visible_count, self.supported_count)

    @property
    def citation_rate(self) -> float:
        return _rate(self.cited_count, self.supported_count)

    @property
    def alignment_rate(self) -> float:
        return _rate(self.aligned_count, self.supported_count)

    @property
    def harmful_rate(self) -> float:
        return _rate(self.harmful_count, self.supported_count)


def summarize_module_attribution(
    traces: list[EvaluationTrace],
) -> dict[str, ModuleAttributionSummary]:
    """Aggregate module exposure attribution metrics by module name."""
    mutable: dict[str, dict[str, int]] = {}
    for trace in traces:
        for exposure in trace.module_exposures:
            module = exposure.module
            if not module:
                continue
            stats = mutable.setdefault(
                module,
                {
                    "exposure_count": 0,
                    "supported_count": 0,
                    "unsupported_count": 0,
                    "prompt_visible_count": 0,
                    "cited_count": 0,
                    "aligned_count": 0,
                    "harmful_count": 0,
                },
            )
            stats["exposure_count"] += 1
            if exposure.support == MetricSupport.UNSUPPORTED:
                stats["unsupported_count"] += 1
            else:
                stats["supported_count"] += 1
            if exposure.prompt_visible:
                stats["prompt_visible_count"] += 1
            if exposure.cited_by_decision:
                stats["cited_count"] += 1
            if exposure.aligned_with_decision:
                stats["aligned_count"] += 1
            if _is_harmful_transfer(exposure):
                stats["harmful_count"] += 1

    return {
        module: ModuleAttributionSummary(module=module, **stats)
        for module, stats in sorted(mutable.items())
    }


def _is_harmful_transfer(exposure: ModuleExposure) -> bool:
    return (
        exposure.metadata.get("harmful_transfer") is True
        or exposure.metadata.get("harmful") is True
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
