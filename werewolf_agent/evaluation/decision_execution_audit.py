# -*- coding: utf-8 -*-
"""
汇总模型决策执行记录的 taxonomy、重试与推理证据指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    ReasoningLevel,
    ReasoningStatus,
    RouteKind,
)
from werewolf_agent.model_gateway.reasoning_policy import (
    minimum_reasoning_level,
    reasoning_capability_satisfies,
)
from werewolf_agent.runtime.decision_outcomes import (
    DecisionOutcome,
    normalize_decision_execution_trace,
    summarize_attempt_counts,
    translate_decision_outcome,
    translate_serialized_decision_outcome,
)

def compute_decision_execution_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """通过唯一 translator 汇总请求级 taxonomy，不解析错误自由文本。"""
    root_causes: Counter[str] = Counter()
    attempt_outcomes: Counter[str] = Counter()
    decision_outcomes: Counter[str] = Counter()
    attempt_count = 0
    retry_count = 0
    provider_fallback_count = 0
    runtime_timeout_count = 0
    decision_count = 0
    invalid_sequence_count = 0
    consistency_errors = 0
    requested_count = 0
    confirmed_count = 0
    unconfirmed_count = 0
    fallback_disabled_count = 0
    fallback_request_count = 0
    fallback_keep_count = 0
    critical_request_count = 0
    critical_request_covered_count = 0
    critical_effective_request_count = 0
    critical_confirmed_count = 0
    critical_confirmation_review_count = 0
    critical_unsupported_count = 0
    critical_not_requested_count = 0
    critical_fallback_disabled_count = 0
    missing_task_type_count = 0
    unknown_task_type_count = 0

    for record in _iter_action_trace_records(games):
        trace = record["trace"]
        runtime_timeout_count_explicit = "runtime_timeout_count" in trace
        timeout_consistency_checked = False
        task_type = record.get("explicit_task_type")
        minimum_level: ReasoningLevel | None = None
        if task_type:
            try:
                candidate_minimum = minimum_reasoning_level(str(task_type))
            except ValueError:
                candidate_minimum = None
                unknown_task_type_count += 1
            if candidate_minimum is not ReasoningLevel.NONE:
                minimum_level = candidate_minimum
        else:
            missing_task_type_count += 1
        raw_attempts = trace.get("execution_attempts")
        if not isinstance(raw_attempts, (list, tuple)) or not raw_attempts:
            supplied_timeout_count = trace.get("runtime_timeout_count")
            if (
                runtime_timeout_count_explicit
                and type(supplied_timeout_count) is int
                and supplied_timeout_count >= 0
                and supplied_timeout_count != 0
            ):
                consistency_errors += 1
            decision_count += 1
            invalid_sequence_count += 1
            if minimum_level is not None:
                # 缺失 attempt 本身就是一个未覆盖的关键决策，必须进入分母。
                critical_request_count += 1
            continue
        try:
            if all(isinstance(item, AttemptExecutionRecord) for item in raw_attempts):
                translated = translate_decision_outcome(tuple(raw_attempts))
            elif all(isinstance(item, Mapping) for item in raw_attempts):
                translated = translate_serialized_decision_outcome(raw_attempts)
            else:
                raise TypeError("execution attempts must share one schema")
        except (KeyError, TypeError, ValueError):
            decision_count += 1
            invalid_sequence_count += 1
            if minimum_level is not None:
                critical_request_count += 1
            continue

        derived_timeout_count = summarize_attempt_counts(
            translated.attempts
        ).runtime_timeout_count
        try:
            normalized_trace = normalize_decision_execution_trace(trace)
        except (KeyError, TypeError, ValueError):
            supplied_timeout_count = trace.get("runtime_timeout_count")
            if (
                not runtime_timeout_count_explicit
                or type(supplied_timeout_count) is not int
                or supplied_timeout_count < 0
            ):
                decision_count += 1
                invalid_sequence_count += 1
                if minimum_level is not None:
                    critical_request_count += 1
                continue
            normalized_trace = dict(trace)
            if not {
                "attempt_count",
                "provider_fallback_count",
                "generated_by",
                "terminal_failure_code",
            }.issubset(trace):
                normalized_trace["normalized_from_schema_version"] = "1"
            consistency_errors += int(
                supplied_timeout_count
                != derived_timeout_count
            )
            timeout_consistency_checked = True

        decision_count += 1
        if minimum_level is not None:
            critical_request_count += len(translated.attempts)
            critical_request_covered_count += sum(
                reasoning_capability_satisfies(
                    attempt.requested_reasoning_level.value,
                    minimum_level.value,
                )
                for attempt in translated.attempts
            )
            for attempt in translated.attempts:
                level_satisfies = reasoning_capability_satisfies(
                    attempt.requested_reasoning_level.value,
                    minimum_level.value,
                )
                status = attempt.normalized_reasoning_status
                if level_satisfies and status in {
                    ReasoningStatus.CONFIRMED,
                    ReasoningStatus.REQUESTED_UNCONFIRMED,
                }:
                    critical_effective_request_count += 1
                if level_satisfies and status is ReasoningStatus.CONFIRMED:
                    critical_confirmed_count += 1
                if (
                    level_satisfies
                    and status is ReasoningStatus.REQUESTED_UNCONFIRMED
                ):
                    critical_confirmation_review_count += 1
                critical_unsupported_count += int(
                    status is ReasoningStatus.UNSUPPORTED
                )
                critical_not_requested_count += int(
                    status is ReasoningStatus.NOT_REQUESTED
                )
                critical_fallback_disabled_count += int(
                    status is ReasoningStatus.FALLBACK_DISABLED
                )
        decision_outcomes[translated.outcome.value] += 1
        attempt_count += translated.attempt_count
        retry_count += translated.retry_count
        provider_fallback_count += translated.provider_fallback_count
        runtime_timeout_count += derived_timeout_count
        for attempt in translated.attempts:
            root_causes[attempt.root_cause.value] += 1
            attempt_outcomes[attempt.attempt_outcome.value] += 1
            if attempt.requested_reasoning_level is not ReasoningLevel.NONE:
                requested_count += 1
                if attempt.normalized_reasoning_status is ReasoningStatus.CONFIRMED:
                    confirmed_count += 1
                if (
                    attempt.normalized_reasoning_status
                    is ReasoningStatus.REQUESTED_UNCONFIRMED
                ):
                    unconfirmed_count += 1
                if (
                    attempt.normalized_reasoning_status
                    is ReasoningStatus.FALLBACK_DISABLED
                ):
                    fallback_disabled_count += 1

        expected_fields = {
            "attempt_count": translated.attempt_count,
            "retry_count": translated.retry_count,
            "provider_fallback_count": translated.provider_fallback_count,
        }
        if "normalized_from_schema_version" not in normalized_trace:
            consistency_errors += sum(
                trace.get(key) != value
                for key, value in expected_fields.items()
            )
        if runtime_timeout_count_explicit and not timeout_consistency_checked:
            supplied_timeout_count = trace.get("runtime_timeout_count")
            if (
                type(supplied_timeout_count) is int
                and supplied_timeout_count >= 0
                and supplied_timeout_count != derived_timeout_count
            ):
                consistency_errors += 1

        has_provider_fallback = any(
            attempt.route_kind is RouteKind.PROVIDER_FALLBACK
            for attempt in translated.attempts
        )
        if (
            has_provider_fallback
            and translated.outcome is not DecisionOutcome.TERMINAL_FALLBACK
            and minimum_level is not None
        ):
            fallback_request_count += 1
            final = translated.final_attempt
            if (
                reasoning_capability_satisfies(
                    final.requested_reasoning_level.value,
                    minimum_level.value,
                )
                and final.normalized_reasoning_status
                not in {ReasoningStatus.UNSUPPORTED, ReasoningStatus.FALLBACK_DISABLED}
            ):
                fallback_keep_count += 1

    return {
        "decision_execution_metrics_supported": decision_count > 0,
        "decision_count": decision_count,
        "attempt_count": attempt_count,
        "retry_count": retry_count,
        "provider_fallback_count": provider_fallback_count,
        "runtime_timeout_count": runtime_timeout_count,
        "root_cause_counts": dict(sorted(root_causes.items())),
        "attempt_outcome_counts": dict(sorted(attempt_outcomes.items())),
        "decision_outcome_counts": dict(sorted(decision_outcomes.items())),
        "decision_execution_invalid_sequence_count": invalid_sequence_count,
        "attempt_retry_consistency_error_count": consistency_errors,
        "reasoning_requested_count": requested_count,
        "reasoning_confirmed_count": confirmed_count,
        "reasoning_unconfirmed_count": unconfirmed_count,
        "reasoning_fallback_disabled_count": fallback_disabled_count,
        "reasoning_request_rate": (
            requested_count / attempt_count if attempt_count else None
        ),
        "reasoning_confirmation_rate": (
            confirmed_count / requested_count if requested_count else None
        ),
        "reasoning_confirmation_supported": requested_count > 0,
        "reasoning_fallback_keep_metrics_supported": fallback_request_count > 0,
        "reasoning_fallback_request_count": fallback_request_count,
        "reasoning_fallback_keep_count": fallback_keep_count,
        "reasoning_fallback_keep_rate": (
            fallback_keep_count / fallback_request_count
            if fallback_request_count else None
        ),
        "critical_task_reasoning_request_coverage_supported": (
            critical_request_count > 0
        ),
        "critical_task_reasoning_request_count": critical_request_count,
        "critical_task_reasoning_requested_count": critical_request_covered_count,
        "critical_task_reasoning_request_coverage": (
            critical_request_covered_count / critical_request_count
            if critical_request_count else None
        ),
        "critical_task_reasoning_effective_request_metrics_supported": (
            critical_request_count > 0
        ),
        "critical_task_reasoning_effective_request_count": (
            critical_effective_request_count
        ),
        "critical_task_reasoning_effective_request_rate": (
            critical_effective_request_count / critical_request_count
            if critical_request_count else None
        ),
        "critical_task_reasoning_request_dispatched_metrics_supported": (
            critical_request_count > 0
        ),
        "critical_task_reasoning_request_dispatched_count": (
            critical_effective_request_count
        ),
        "critical_task_reasoning_request_dispatched_rate": (
            critical_effective_request_count / critical_request_count
            if critical_request_count else None
        ),
        "critical_task_reasoning_confirmed_count": critical_confirmed_count,
        "critical_task_reasoning_confirmed_rate": (
            critical_confirmed_count / critical_request_count
            if critical_request_count else None
        ),
        "critical_task_reasoning_requires_confirmation_review_count": (
            critical_confirmation_review_count
        ),
        "critical_task_reasoning_unsupported_count": critical_unsupported_count,
        "critical_task_reasoning_not_requested_count": critical_not_requested_count,
        "critical_task_reasoning_fallback_disabled_count": (
            critical_fallback_disabled_count
        ),
        "reasoning_task_type_missing_count": missing_task_type_count,
        "reasoning_task_type_unknown_count": unknown_task_type_count,
        **_critical_reasoning_status_metrics(games),
    }


def _critical_reasoning_status_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """统计关键玩家请求是否为每个 attempt 显式记录推理状态。"""
    total = 0
    explicit = 0
    for record in _iter_action_trace_records(games):
        task_type = record.get("explicit_task_type")
        if not task_type:
            continue
        try:
            minimum = minimum_reasoning_level(str(task_type))
        except ValueError:
            continue
        if minimum is ReasoningLevel.NONE:
            continue
        total += 1
        attempts = record["trace"].get("execution_attempts")
        if not isinstance(attempts, (list, tuple)) or not attempts:
            continue
        statuses: list[Any] = []
        for attempt in attempts:
            if isinstance(attempt, AttemptExecutionRecord):
                statuses.append(attempt.normalized_reasoning_status)
            elif isinstance(attempt, Mapping):
                statuses.append(attempt.get("normalized_reasoning_status"))
            else:
                statuses.append(None)
        if all(
            isinstance(status, ReasoningStatus)
            or status in {item.value for item in ReasoningStatus}
            for status in statuses
        ):
            explicit += 1
    return {
        "critical_task_reasoning_status_metrics_supported": total > 0,
        "critical_task_reasoning_status_request_count": total,
        "critical_task_reasoning_status_explicit_count": explicit,
        "critical_task_reasoning_status_explicit_rate": (
            explicit / total if total else None
        ),
    }


def _iter_action_traces(games: list[dict[str, Any]]):
    for record in _iter_action_trace_records(games):
        yield record["trace"]


def _iter_action_trace_records(games: list[dict[str, Any]]):
    for game in games:
        for event in game.get("events", []):
            payload = event.get("payload") or {}
            trace = payload.get("action_trace")
            if isinstance(trace, Mapping):
                yield {
                    "trace": trace,
                    "actor": _trace_actor(payload, trace),
                    "task": _trace_task(payload, trace, event.get("type")),
                    "explicit_task_type": _explicit_trace_task(payload, trace),
                    "game": game,
                }
            elif (
                event.get("type") == "action_trace_audit"
                and _critical_explicit_task(payload)
            ):
                # 审计事件已声明关键任务却没有合法 trace 时仍要进入失败分母。
                yield {
                    "trace": {},
                    "actor": _trace_actor(payload, {}),
                    "task": _trace_task(payload, {}, event.get("type")),
                    "explicit_task_type": _explicit_trace_task(payload, {}),
                    "game": game,
                }
            traces = payload.get("action_traces")
            if isinstance(traces, Mapping):
                for actor_id, item in traces.items():
                    if isinstance(item, Mapping):
                        yield {
                            "trace": item,
                            "actor": _trace_actor(payload, item) or actor_id,
                            "task": _trace_task(payload, item, event.get("type")),
                            "explicit_task_type": _explicit_trace_task(payload, item),
                            "game": game,
                        }


def _critical_explicit_task(payload: dict[str, Any]) -> bool:
    task_type = _explicit_trace_task(payload, {})
    if not task_type:
        return False
    try:
        return minimum_reasoning_level(task_type) is not ReasoningLevel.NONE
    except ValueError:
        return False


def _trace_actor(payload: dict[str, Any], trace: dict[str, Any]) -> Any:
    for source in (trace, payload):
        for key in (
            "agent_id",
            "player_id",
            "actor_id",
            "actor",
            "speaker",
            "voter",
            "wolf_id",
            "seer_id",
            "witch_id",
            "hunter_id",
        ):
            actor = source.get(key)
            if actor:
                return actor
    return None


def _trace_task(
    payload: dict[str, Any],
    trace: dict[str, Any],
    event_type: Any,
) -> str:
    for source in (trace, payload):
        for key in ("task_type", "phase", "task"):
            value = source.get(key)
            if value:
                return str(value).lower()
    return str(event_type or "").lower()


def _explicit_trace_task(
    payload: dict[str, Any],
    trace: dict[str, Any],
) -> str | None:
    """只读取明确的 task_type，避免把 phase 猜测成推理策略任务。"""
    for source in (trace, payload):
        value = source.get("task_type")
        if value:
            return str(value).lower()
    return None
