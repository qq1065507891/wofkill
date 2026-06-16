"""Offline ablation helpers for feedback-loop evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass, replace

from werewolf_agent.evaluation.feedback_metrics import (
    ModuleAttributionSummary,
    summarize_module_attribution,
)
from werewolf_agent.evaluation.feedback_schemas import EvaluationTrace


OFFLINE_UNSUPPORTED_METRICS: dict[str, str] = {
    "live_win_rate_delta": "offline_trace_mode",
    "causal_decision_delta": "offline_trace_mode",
}


@dataclass(frozen=True)
class AblationReport:
    mode: str
    removed_modules: list[str]
    baseline_trace_count: int
    ablated_trace_count: int
    baseline_module_metrics: dict[str, ModuleAttributionSummary]
    ablated_module_metrics: dict[str, ModuleAttributionSummary]
    unsupported_metrics: dict[str, str]
    ablated_traces: list[EvaluationTrace]


class OfflineTraceAblationRunner:
    """Runs deterministic ablation by removing module exposures from traces."""

    def run(
        self,
        traces: list[EvaluationTrace],
        *,
        removed_modules: list[str],
    ) -> AblationReport:
        modules_to_remove = _normalize_modules(removed_modules)
        ablated_traces = [
            _remove_module_exposures(trace, modules_to_remove)
            for trace in traces
        ]

        return AblationReport(
            mode="offline_trace",
            removed_modules=modules_to_remove,
            baseline_trace_count=len(traces),
            ablated_trace_count=len(ablated_traces),
            baseline_module_metrics=summarize_module_attribution(traces),
            ablated_module_metrics=summarize_module_attribution(ablated_traces),
            unsupported_metrics=dict(OFFLINE_UNSUPPORTED_METRICS),
            ablated_traces=ablated_traces,
        )


def _normalize_modules(modules: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for module in modules:
        name = module.strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _remove_module_exposures(
    trace: EvaluationTrace,
    removed_modules: list[str],
) -> EvaluationTrace:
    if not removed_modules:
        return trace

    blocked = set(removed_modules)
    return replace(
        trace,
        module_exposures=[
            exposure
            for exposure in trace.module_exposures
            if exposure.module not in blocked
        ],
    )
