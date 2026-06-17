"""Regression gate for reviewable improvement candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateRegressionConfig:
    candidate_id: str
    target_faction: str = "good"
    hidden_info_leak_increase_tolerance: float = 0.0
    illegal_action_increase_tolerance: float = 0.0
    vote_quality_drop_tolerance: float = 0.0
    judge_consistency_rate_drop_tolerance: float = 0.0
    target_faction_win_rate_drop_tolerance: float = 0.0
    harmful_transfer_increase_tolerance: float = 0.0
    # 缺失 producer 时让门 fail-closed 而非静默跳过；默认空保持历史行为
    required_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateMetricDelta:
    metric: str
    baseline: float
    candidate: float
    candidate_minus_baseline: float
    higher_is_better: bool
    regression_amount: float
    tolerance: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    reason: str = ""
    metric: str = ""
    regression_amount: float = 0.0
    tolerance: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    blocked_reasons: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRegressionReport:
    candidate_id: str
    passed: bool
    metric_deltas: list[CandidateMetricDelta]
    checks: list[GateCheck]
    blocked_reasons: list[str]
    prompt_safe: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "prompt_safe": self.prompt_safe,
            "blocked_reasons": list(self.blocked_reasons),
            "metric_deltas": [
                delta.to_json_dict()
                for delta in self.metric_deltas
            ],
            "checks": [
                check.to_json_dict()
                for check in self.checks
            ],
        }


class RegressionGate:
    """Compare current-system baseline metrics with candidate draft metrics."""

    def evaluate(
        self,
        config: CandidateRegressionConfig,
        *,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        prompt_safe: bool,
    ) -> CandidateRegressionReport:
        metric_deltas: list[CandidateMetricDelta] = []
        checks: list[GateCheck] = []

        self._add_lower_is_better_check(
            metric_deltas,
            checks,
            name="hidden_info_leak",
            reason="hidden_info_leak_increased",
            metric="hidden_info_leak_rate",
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            tolerance=config.hidden_info_leak_increase_tolerance,
        )
        self._add_lower_is_better_check(
            metric_deltas,
            checks,
            name="illegal_action",
            reason="illegal_action_increased",
            metric=_present_metric(
                baseline_metrics,
                candidate_metrics,
                ("illegal_action_rate", "illegal_action_count"),
            ),
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            tolerance=config.illegal_action_increase_tolerance,
        )
        self._add_higher_is_better_check(
            metric_deltas,
            checks,
            name="vote_quality",
            reason="vote_quality_dropped",
            metric="vote_quality",
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            tolerance=config.vote_quality_drop_tolerance,
        )
        self._add_higher_is_better_check(
            metric_deltas,
            checks,
            name="judge_consistency_rate",
            reason="judge_consistency_rate_dropped",
            metric="judge_consistency_rate",
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            tolerance=config.judge_consistency_rate_drop_tolerance,
        )
        target_win_rate_metric = f"{config.target_faction}_win_rate"
        self._add_higher_is_better_check(
            metric_deltas,
            checks,
            name=target_win_rate_metric,
            reason=f"{target_win_rate_metric}_dropped",
            metric=target_win_rate_metric,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            tolerance=config.target_faction_win_rate_drop_tolerance,
        )
        self._add_lower_is_better_check(
            metric_deltas,
            checks,
            name="harmful_transfer",
            reason="harmful_transfer_increased",
            metric="harmful_transfer_rate",
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            tolerance=config.harmful_transfer_increase_tolerance,
        )

        if not prompt_safe:
            checks.append(GateCheck(
                name="prompt_safety",
                passed=False,
                reason="prompt_safety_failed",
            ))
        else:
            checks.append(GateCheck(name="prompt_safety", passed=True))

        for required in config.required_metrics:
            if required in baseline_metrics or required in candidate_metrics:
                continue
            checks.append(GateCheck(
                name=f"required_{required}",
                passed=False,
                reason=f"required_metric_missing:{required}",
                metric=required,
            ))

        blocked_reasons = [
            check.reason
            for check in checks
            if not check.passed and check.reason
        ]
        return CandidateRegressionReport(
            candidate_id=config.candidate_id,
            passed=not blocked_reasons,
            metric_deltas=metric_deltas,
            checks=checks,
            blocked_reasons=blocked_reasons,
            prompt_safe=prompt_safe,
        )

    def _add_lower_is_better_check(
        self,
        metric_deltas: list[CandidateMetricDelta],
        checks: list[GateCheck],
        *,
        name: str,
        reason: str,
        metric: str,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        tolerance: float,
    ) -> None:
        if not metric:
            return
        if metric not in baseline_metrics and metric not in candidate_metrics:
            return
        delta = _metric_delta(
            metric,
            baseline_metrics,
            candidate_metrics,
            higher_is_better=False,
            tolerance=tolerance,
        )
        metric_deltas.append(delta)
        passed = delta.regression_amount <= tolerance
        checks.append(GateCheck(
            name=name,
            passed=passed,
            reason="" if passed else reason,
            metric=metric,
            regression_amount=delta.regression_amount,
            tolerance=tolerance,
        ))

    def _add_higher_is_better_check(
        self,
        metric_deltas: list[CandidateMetricDelta],
        checks: list[GateCheck],
        *,
        name: str,
        reason: str,
        metric: str,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        tolerance: float,
    ) -> None:
        if metric not in baseline_metrics and metric not in candidate_metrics:
            return
        delta = _metric_delta(
            metric,
            baseline_metrics,
            candidate_metrics,
            higher_is_better=True,
            tolerance=tolerance,
        )
        metric_deltas.append(delta)
        passed = delta.regression_amount <= tolerance
        checks.append(GateCheck(
            name=name,
            passed=passed,
            reason="" if passed else reason,
            metric=metric,
            regression_amount=delta.regression_amount,
            tolerance=tolerance,
        ))


def _metric_delta(
    metric: str,
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    *,
    higher_is_better: bool,
    tolerance: float,
) -> CandidateMetricDelta:
    baseline = float(baseline_metrics.get(metric, 0.0))
    candidate = float(candidate_metrics.get(metric, 0.0))
    candidate_minus_baseline = _stable_float(candidate - baseline)
    if higher_is_better:
        regression_amount = _stable_float(max(0.0, baseline - candidate))
    else:
        regression_amount = _stable_float(max(0.0, candidate - baseline))
    return CandidateMetricDelta(
        metric=metric,
        baseline=baseline,
        candidate=candidate,
        candidate_minus_baseline=candidate_minus_baseline,
        higher_is_better=higher_is_better,
        regression_amount=regression_amount,
        tolerance=tolerance,
    )


def _stable_float(value: float) -> float:
    return round(float(value), 10)


def _present_metric(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    names: tuple[str, ...],
) -> str:
    for name in names:
        if name in baseline_metrics or name in candidate_metrics:
            return name
    return ""


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value
