"""Offline ablation helpers for feedback-loop evaluation traces."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

from werewolf_agent.agents.schemas import AgentContext, FallbackAction, PlayerAction
from werewolf_agent.evaluation.feedback_metrics import (
    ModuleAttributionSummary,
    summarize_module_attribution,
)
from werewolf_agent.evaluation.feedback_schemas import DecisionSnapshot, EvaluationTrace


OFFLINE_UNSUPPORTED_METRICS: dict[str, str] = {
    "live_win_rate_delta": "offline_trace_mode",
    "causal_decision_delta": "offline_trace_mode",
}
LIVE_CONTEXT_SUPPORTED_TOGGLES = {
    "rag",
    "reflection",
    "possible_worlds",
    "simulator",
    "skills",
    "persona",
}
LIVE_CONTEXT_UNSUPPORTED_METRICS: dict[str, str] = {
    "live_win_rate_delta": "live_context_mode_decision_only",
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


@dataclass(frozen=True)
class AblationToggleSet:
    removed_modules: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "removed_modules",
            _normalize_modules(self.removed_modules),
        )


@dataclass(frozen=True)
class LiveAblationDecision:
    action_type: str
    target_id: str | None
    confidence: float
    raw: dict[str, Any]
    error: str = ""


@dataclass(frozen=True)
class LiveAblationPair:
    context_id: str
    baseline: LiveAblationDecision
    ablated: LiveAblationDecision
    action_changed: bool
    target_changed: bool
    confidence_delta: float


@dataclass(frozen=True)
class LiveAblationReport:
    mode: str
    removed_modules: list[str]
    pair_count: int
    failed_pair_count: int
    action_changed_count: int
    target_changed_count: int
    avg_confidence_delta: float
    unsupported_metrics: dict[str, str]
    pairs: list[LiveAblationPair]


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


class LiveContextAblationHarness:
    """Compares injected baseline and ablated decisions for live contexts."""

    def __init__(self, runner: Callable[[AgentContext], Any]) -> None:
        self._runner = runner

    def run(
        self,
        contexts: list[AgentContext],
        *,
        toggles: AblationToggleSet,
    ) -> LiveAblationReport:
        pairs: list[LiveAblationPair] = []
        valid_pairs: list[LiveAblationPair] = []

        for index, context in enumerate(contexts):
            baseline = self._run_one(context)
            ablated_context = apply_ablation_toggles(context, toggles)
            ablated = self._run_one(ablated_context)

            failed = bool(baseline.error or ablated.error)
            pair = LiveAblationPair(
                context_id=_context_id(index, context),
                baseline=baseline,
                ablated=ablated,
                action_changed=(
                    not failed and baseline.action_type != ablated.action_type
                ),
                target_changed=(
                    not failed and baseline.target_id != ablated.target_id
                ),
                confidence_delta=(
                    0.0 if failed else ablated.confidence - baseline.confidence
                ),
            )
            pairs.append(pair)
            if not failed:
                valid_pairs.append(pair)

        unsupported_metrics = dict(LIVE_CONTEXT_UNSUPPORTED_METRICS)
        for module in toggles.removed_modules:
            if module not in LIVE_CONTEXT_SUPPORTED_TOGGLES:
                unsupported_metrics[module] = f"unsupported_toggle:{module}"

        avg_confidence_delta = (
            sum(pair.confidence_delta for pair in valid_pairs) / len(valid_pairs)
            if valid_pairs
            else 0.0
        )

        return LiveAblationReport(
            mode="live_context",
            removed_modules=toggles.removed_modules,
            pair_count=len(pairs),
            failed_pair_count=sum(
                1 for pair in pairs if pair.baseline.error or pair.ablated.error
            ),
            action_changed_count=sum(1 for pair in valid_pairs if pair.action_changed),
            target_changed_count=sum(1 for pair in valid_pairs if pair.target_changed),
            avg_confidence_delta=avg_confidence_delta,
            unsupported_metrics=unsupported_metrics,
            pairs=pairs,
        )

    def _run_one(self, context: AgentContext) -> LiveAblationDecision:
        try:
            return _to_live_ablation_decision(self._runner(context))
        except Exception as exc:
            return LiveAblationDecision(
                action_type="",
                target_id=None,
                confidence=0.0,
                raw={},
                error=str(exc),
            )


def apply_ablation_toggles(
    context: AgentContext,
    toggles: AblationToggleSet,
) -> AgentContext:
    update: dict[str, object] = {}
    removed = set(toggles.removed_modules)

    if "rag" in removed:
        update["rag_hints"] = []
    if "reflection" in removed:
        update["reflection_memory_hints"] = []
        update["error_pattern_hint"] = {}
    if "possible_worlds" in removed:
        update["possible_worlds"] = {}
    if "simulator" in removed:
        update["simulation_predictions"] = {}
    if "skills" in removed:
        strategy = copy.deepcopy(context.strategy_directive)
        strategy.pop("skill_tactical_advice", None)
        update["strategy_directive"] = strategy
        update["skill_analyses"] = {}
        update["skill_analysis_hints"] = {}
    if "persona" in removed:
        update["persona_snapshot"] = {}

    return context.model_copy(deep=True, update=update)


def _context_id(index: int, context: AgentContext) -> str:
    task_type = _value(context.task_type)
    return (
        f"{index}:{context.agent_id}:{task_type}:{context.phase}:"
        f"D{context.day_number}:N{context.night_number}"
    )


def _to_live_ablation_decision(output: Any) -> LiveAblationDecision:
    if isinstance(output, PlayerAction):
        return LiveAblationDecision(
            action_type=_value(output.action_type),
            target_id=output.target_id,
            confidence=float(output.confidence),
            raw=_model_dump(output),
        )
    if isinstance(output, FallbackAction):
        return LiveAblationDecision(
            action_type=_value(output.action_type),
            target_id=output.target_id,
            confidence=0.0,
            raw=_model_dump(output),
        )
    if isinstance(output, DecisionSnapshot):
        return LiveAblationDecision(
            action_type=output.action_type,
            target_id=output.target_id,
            confidence=float(output.confidence),
            raw=dict(output.raw),
        )
    if isinstance(output, dict):
        return LiveAblationDecision(
            action_type=str(output.get("action_type", "")),
            target_id=output.get("target_id"),
            confidence=float(output.get("confidence", 0.0) or 0.0),
            raw=dict(output),
        )

    action_type = _value(getattr(output, "action_type", ""))
    return LiveAblationDecision(
        action_type=action_type,
        target_id=getattr(output, "target_id", None),
        confidence=float(getattr(output, "confidence", 0.0) or 0.0),
        raw={},
    )


def _model_dump(model: PlayerAction | FallbackAction) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


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
