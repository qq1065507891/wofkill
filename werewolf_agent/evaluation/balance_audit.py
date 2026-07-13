# -*- coding: utf-8 -*-
"""
功能描述：消费已保存的 JSON 对局和强类型运行时审计，汇总平衡与验收指标。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-13
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from werewolf_agent.evaluation.balance_public_claims import (
    night_info_claim_supported as _night_info_claim_supported,  # noqa: F401
    role_claim_supported as _role_claim_supported,  # noqa: F401
    unsupported_claims_in_text as _unsupported_claims_in_text,  # noqa: F401
    unsupported_public_fact_claim_count as _unsupported_public_fact_claim_count,
)
from werewolf_agent.runtime.exposure_audit import summarize_persona_prompt_confirmation
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
    translate_decision_outcome,
    translate_serialized_decision_outcome,
)

_FAILURE_TRACE_FIELDS = ("fallback_reason", "parse_error", "structured_failure_reason")
_POWER_ROLES = {"seer", "witch", "hunter", "idiot"}
_TEMPLATE_VOTE_REASON_MARKERS = (
    "当前合法投票候选",
    "继续施压",
)


def load_game_logs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load saved game JSON logs from disk."""
    games: list[dict[str, Any]] = []
    for path in paths:
        games.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return games


def compute_balance_audit(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute balance and quality metrics from saved game logs."""
    game_count = len(games)
    wolf_wins = sum(1 for game in games if game.get("winning_faction") == "werewolf")
    good_wins = sum(1 for game in games if game.get("winning_faction") == "good")

    action_trace_records = list(_iter_action_trace_records(games))
    action_traces = [record["trace"] for record in action_trace_records]
    fallback_count = sum(1 for trace in action_traces if trace.get("fallback_reason"))
    schema_failures = sum(
        1 for trace in action_traces
        if trace.get("parse_error") or trace.get("structured_failure_reason")
    )
    wolf_plan_outcomes = compute_wolf_plan_outcome_metrics(games)
    wolf_plan_fallback_count = wolf_plan_outcomes[
        "wolf_team_plan_terminal_fallback_count"
    ]
    wolf_plan_count = wolf_plan_outcomes["wolf_team_plan_total_count"]

    weak_wolf_plan_kill_count = sum(_weak_wolf_plan_kills(game) for game in games)
    fallback_plan_kill_without_target_evidence_count = sum(
        _fallback_plan_kill_without_target_evidence_count(game)
        for game in games
    )
    fallback_plan_kill_count = sum(_fallback_plan_kill_count(game) for game in games)
    vote_concentrations = [_vote_concentration(event) for game in games for event in game.get("events", []) if event.get("type") == "vote_resolved"]
    warnings: list[str] = []

    wolf_win_rate = wolf_wins / game_count if game_count else 0.0
    good_win_rate = good_wins / game_count if game_count else 0.0
    fallback_action_rate = fallback_count / len(action_traces) if action_traces else 0.0
    wolf_team_plan_fallback_rate = (
        wolf_plan_fallback_count / wolf_plan_count
        if wolf_plan_count else None
    )
    schema_failure_rate = schema_failures / len(action_traces) if action_traces else 0.0
    seer_day1_exile_rate = _seer_day1_exile_rate(games)
    witch_night1_death_rate = _witch_night1_death_rate(games)
    witch_wolf_kill_death_rate = _witch_wolf_kill_death_rate(games)
    sheriff_werewolf_rate = _sheriff_werewolf_rate(games)
    sheriff_vote_fallback_rate = _sheriff_vote_fallback_rate(action_trace_records)
    hunter_friendly_fire_rate = _hunter_friendly_fire_rate(games)
    weak_plan_kill_rate = _weak_plan_kill_rate(games)
    fallback_plan_kill_without_target_evidence_rate = (
        fallback_plan_kill_without_target_evidence_count / fallback_plan_kill_count
        if fallback_plan_kill_count else 0.0
    )
    power_role_fallback_rate = _power_role_fallback_rate(action_trace_records)
    template_vote_reason_count, vote_reason_count = _template_vote_reason_counts(games)
    template_vote_reason_rate = (
        template_vote_reason_count / vote_reason_count
        if vote_reason_count else 0.0
    )
    unsupported_public_fact_claim_count = sum(
        _unsupported_public_fact_claim_count(game) for game in games
    )
    persona_prompt_confirmation = summarize_persona_prompt_confirmation([
        event
        for game in games
        for event in game.get("events", [])
        if isinstance(event, dict)
    ])
    acceptance_metrics = compute_acceptance_audit_metrics(games)
    mean_vote_concentration = (
        sum(vote_concentrations) / len(vote_concentrations)
        if vote_concentrations else 0.0
    )

    if game_count >= 3 and wolf_win_rate > 0.75:
        warnings.append("wolf_win_rate_high")
    if schema_failure_rate > 0.05:
        warnings.append("schema_failure_high")
    if (
        wolf_team_plan_fallback_rate is not None
        and wolf_team_plan_fallback_rate > 0.5
    ):
        warnings.append("wolf_team_plan_fallback_high")
    if game_count >= 3 and seer_day1_exile_rate > 0.35:
        warnings.append("seer_day1_exile_high")
    if weak_wolf_plan_kill_count:
        warnings.append("weak_wolf_plan_kills_present")
    if sheriff_werewolf_rate > 0.6:
        warnings.append("sheriff_werewolf_rate_high")
    if sheriff_vote_fallback_rate > 0.2:
        warnings.append("sheriff_vote_fallback_high")
    if hunter_friendly_fire_rate > 0.5:
        warnings.append("hunter_friendly_fire_high")
    if weak_plan_kill_rate > 0.2:
        warnings.append("weak_plan_kill_high")
    if fallback_plan_kill_without_target_evidence_count:
        warnings.append("fallback_plan_kill_without_target_evidence_present")
    if template_vote_reason_rate > 0.1:
        warnings.append("template_vote_reason_high")
    if unsupported_public_fact_claim_count:
        warnings.append("unsupported_public_fact_claims_present")

    return {
        "games": game_count,
        "wolf_win_rate": wolf_win_rate,
        "good_win_rate": good_win_rate,
        "fallback_action_rate": fallback_action_rate,
        "wolf_team_plan_fallback_rate": wolf_team_plan_fallback_rate,
        "wolf_team_plan_fallback_count": wolf_plan_fallback_count,
        "wolf_team_plan_count": wolf_plan_count,
        **wolf_plan_outcomes,
        "schema_failure_rate": schema_failure_rate,
        "seer_day1_exile_rate": seer_day1_exile_rate,
        "d1_seer_exile_rate": seer_day1_exile_rate,
        "witch_night1_death_rate": witch_night1_death_rate,
        "witch_wolf_kill_death_rate": witch_wolf_kill_death_rate,
        "sheriff_werewolf_rate": sheriff_werewolf_rate,
        "sheriff_vote_fallback_rate": sheriff_vote_fallback_rate,
        "hunter_friendly_fire_rate": hunter_friendly_fire_rate,
        "weak_plan_kill_rate": weak_plan_kill_rate,
        "fallback_plan_kill_without_target_evidence_rate": (
            fallback_plan_kill_without_target_evidence_rate
        ),
        "fallback_plan_kill_without_target_evidence_count": (
            fallback_plan_kill_without_target_evidence_count
        ),
        "power_role_fallback_rate": power_role_fallback_rate,
        "mean_vote_concentration": mean_vote_concentration,
        "template_vote_reason_count": template_vote_reason_count,
        "template_vote_reason_rate": template_vote_reason_rate,
        "unsupported_public_fact_claim_count": unsupported_public_fact_claim_count,
        "weak_wolf_plan_kill_count": weak_wolf_plan_kill_count,
        "persona_prompt_confirmation": persona_prompt_confirmation,
        "persona_injection_confirmation_metrics_supported": (
            persona_prompt_confirmation["supported"]
        ),
        "persona_injection_confirmation_rate": (
            persona_prompt_confirmation["confirmation_rate"]
        ),
        **acceptance_metrics,
        "warnings": warnings,
    }


def compute_decision_execution_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """通过唯一 translator 汇总请求级 taxonomy，不解析错误自由文本。"""
    root_causes: Counter[str] = Counter()
    attempt_outcomes: Counter[str] = Counter()
    decision_outcomes: Counter[str] = Counter()
    attempt_count = 0
    retry_count = 0
    decision_count = 0
    invalid_sequence_count = 0
    consistency_errors = 0
    requested_count = 0
    confirmed_count = 0
    fallback_disabled_count = 0
    fallback_request_count = 0
    fallback_keep_count = 0
    critical_request_count = 0
    critical_request_covered_count = 0
    missing_task_type_count = 0

    for record in _iter_action_trace_records(games):
        trace = record["trace"]
        raw_attempts = trace.get("execution_attempts")
        if not isinstance(raw_attempts, (list, tuple)) or not raw_attempts:
            continue
        try:
            if all(isinstance(item, AttemptExecutionRecord) for item in raw_attempts):
                translated = translate_decision_outcome(tuple(raw_attempts))
            elif all(isinstance(item, dict) for item in raw_attempts):
                translated = translate_serialized_decision_outcome(raw_attempts)
            else:
                raise TypeError("execution attempts must share one schema")
        except (KeyError, TypeError, ValueError):
            invalid_sequence_count += 1
            continue

        decision_count += 1
        task_type = record.get("explicit_task_type")
        minimum_level: ReasoningLevel | None = None
        if task_type:
            try:
                candidate_minimum = minimum_reasoning_level(str(task_type))
            except ValueError:
                candidate_minimum = None
            if candidate_minimum is not ReasoningLevel.NONE:
                minimum_level = candidate_minimum
        else:
            missing_task_type_count += 1
        first_attempt = translated.attempts[0]
        if minimum_level is not None:
            critical_request_count += 1
            if reasoning_capability_satisfies(
                first_attempt.requested_reasoning_level.value,
                minimum_level.value,
            ):
                critical_request_covered_count += 1
        decision_outcomes[translated.outcome.value] += 1
        attempt_count += len(translated.attempts)
        retry_count += translated.retry_count
        for attempt in translated.attempts:
            root_causes[attempt.root_cause.value] += 1
            attempt_outcomes[attempt.attempt_outcome.value] += 1
            if attempt.requested_reasoning_level is not ReasoningLevel.NONE:
                requested_count += 1
                if attempt.normalized_reasoning_status is ReasoningStatus.CONFIRMED:
                    confirmed_count += 1
                if (
                    attempt.normalized_reasoning_status
                    is ReasoningStatus.FALLBACK_DISABLED
                ):
                    fallback_disabled_count += 1

        expected_retry_count = len(translated.attempts) - 1
        if trace.get("retry_count") is not None:
            consistency_errors += int(trace["retry_count"] != expected_retry_count)
        expected_success_retries = (
            None
            if translated.outcome is DecisionOutcome.TERMINAL_FALLBACK
            else translated.retry_count
        )
        if "total_retry_count_until_success" in trace:
            consistency_errors += int(
                trace["total_retry_count_until_success"] != expected_success_retries
            )

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

    unconfirmed_count = requested_count - confirmed_count - fallback_disabled_count
    return {
        "decision_execution_metrics_supported": decision_count > 0,
        "decision_count": decision_count,
        "attempt_count": attempt_count,
        "retry_count": retry_count,
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
        "reasoning_task_type_missing_count": missing_task_type_count,
    }


def compute_acceptance_audit_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """组合执行 taxonomy 与跨责任域验收指标，供单局和批量报告复用。"""
    return {
        **compute_decision_execution_metrics(games),
        **_compute_acceptance_metrics(games),
    }


def _compute_acceptance_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总不依赖模型调用的最终验收指标。"""
    semantic_rows: list[dict[str, Any]] = []
    world_groups: list[list[dict[str, Any]]] = []
    power_decisions: list[dict[str, Any]] = []
    post_win_calls = 0
    rejected_facts = 0
    rejected_lessons = 0

    for game in games:
        events = game.get("events", [])
        victory_seen = False
        latest_reflections: dict[str, tuple[int, dict[str, Any]]] = {}
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "victory":
                victory_seen = True
                continue
            if victory_seen and event_type in {
                "action_trace_audit",
                "model_execution_audit",
                "wolf_team_plan",
            }:
                task = str(payload.get("task_type") or payload.get("phase") or "")
                if task not in {"reflection", "post_game_reflection"}:
                    post_win_calls += 1
            if event_type == "semantic_repair_audit" and payload.get("repairable") is True:
                semantic_rows.append(payload)
            if event_type == "reflection_complete":
                for entry in payload.get("entries", []):
                    if not isinstance(entry, dict):
                        continue
                    player_id = entry.get("player_id")
                    verification = entry.get("verification")
                    if isinstance(player_id, str) and isinstance(verification, dict):
                        latest_reflections[player_id] = (event_index, verification)
            if event_type != "action_trace_audit":
                continue
            trace = payload.get("action_trace")
            if not isinstance(trace, dict):
                continue
            audit = trace.get("world_model_audit")
            possible = audit.get("possible_worlds") if isinstance(audit, dict) else None
            top_worlds = possible.get("top_worlds") if isinstance(possible, dict) else None
            if isinstance(top_worlds, list):
                world_groups.append([
                    item for item in top_worlds if isinstance(item, dict)
                ])
            if trace.get("final_action_type") in {"hunter_shot", "use_poison"}:
                evidence = trace.get("power_role_evidence")
                power_decisions.append(evidence if isinstance(evidence, dict) else {})
        for _, verification in latest_reflections.values():
            rejected_facts += _non_negative_int(verification.get("rejected_fact_count"))
            rejected_lessons += _non_negative_int(
                verification.get("rejected_lesson_count")
            )

    semantic_count = len(semantic_rows)
    semantic_success = sum(
        row.get("success") is True and row.get("target_preserved") is True
        for row in semantic_rows
    )
    target_preserved = sum(row.get("target_preserved") is True for row in semantic_rows)
    no_new_claim = sum(
        _non_negative_int(row.get("introduced_claim_count")) == 0
        for row in semantic_rows
    )
    world_count = sum(len(group) for group in world_groups)
    unique_world_count = sum(
        len({
            json.dumps(
                world.get("key_assignments") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            for world in group
        })
        for group in world_groups
    )
    evidence_covered = sum(
        isinstance(world.get("why"), list)
        and bool(world["why"])
        and all(isinstance(ref, str) and bool(ref) for ref in world["why"])
        for group in world_groups
        for world in group
    )
    complete_power_decisions = sum(
        _power_role_evidence_complete(evidence) for evidence in power_decisions
    )
    power_count = len(power_decisions)
    return {
        "terminal_post_win_game_model_call_count": post_win_calls,
        "semantic_repair_metrics_supported": semantic_count > 0,
        "semantic_repair_eligible_count": semantic_count,
        "semantic_repair_success_count": semantic_success,
        "semantic_repair_success_rate": (
            semantic_success / semantic_count if semantic_count else None
        ),
        "semantic_repair_target_preservation_rate": (
            target_preserved / semantic_count if semantic_count else None
        ),
        "semantic_repair_no_new_claim_rate": (
            no_new_claim / semantic_count if semantic_count else None
        ),
        "possible_world_metrics_supported": world_count > 0,
        "possible_world_prompt_count": len(world_groups),
        "possible_world_total_count": world_count,
        "possible_world_unique_count": unique_world_count,
        "possible_world_unique_rate": (
            unique_world_count / world_count if world_count else None
        ),
        "possible_world_evidence_covered_count": evidence_covered,
        "possible_world_evidence_coverage_rate": (
            evidence_covered / world_count if world_count else None
        ),
        "power_role_evidence_metrics_supported": power_count > 0,
        "power_role_damage_decision_count": power_count,
        "power_role_evidence_complete_count": complete_power_decisions,
        "power_role_evidence_completeness_rate": (
            complete_power_decisions / power_count if power_count else None
        ),
        "reflection_rejected_fact_count": rejected_facts,
        "reflection_rejected_lesson_count": rejected_lessons,
    }


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _power_role_evidence_complete(evidence: dict[str, Any]) -> bool:
    comparison = evidence.get("alternative_comparison")
    risk = evidence.get("friendly_fire_risk")
    retain = evidence.get("retain_option")
    if (
        not evidence.get("target_id")
        or not isinstance(risk, dict)
        or not isinstance(retain, dict)
    ):
        return False
    if not isinstance(comparison, dict):
        return False
    alternatives = comparison.get("legal_alternatives")
    if not isinstance(alternatives, list):
        return False
    if "alternative_target" not in comparison:
        return False
    alternative_target = comparison["alternative_target"]
    if comparison.get("no_legal_alternative") is True:
        return alternative_target is None
    return alternative_target is not None and alternative_target in alternatives


def _iter_action_traces(games: list[dict[str, Any]]):
    for record in _iter_action_trace_records(games):
        yield record["trace"]


def _iter_action_trace_records(games: list[dict[str, Any]]):
    for game in games:
        for event in game.get("events", []):
            payload = event.get("payload") or {}
            trace = payload.get("action_trace")
            if isinstance(trace, dict):
                yield {
                    "trace": trace,
                    "actor": _trace_actor(payload, trace),
                    "task": _trace_task(payload, trace, event.get("type")),
                    "explicit_task_type": _explicit_trace_task(payload, trace),
                    "game": game,
                }
            traces = payload.get("action_traces")
            if isinstance(traces, dict):
                for actor_id, item in traces.items():
                    if isinstance(item, dict):
                        yield {
                            "trace": item,
                            "actor": _trace_actor(payload, item) or actor_id,
                            "task": _trace_task(payload, item, event.get("type")),
                            "explicit_task_type": _explicit_trace_task(payload, item),
                            "game": game,
                        }


def _wolf_plan_fallback_count(game: dict[str, Any]) -> int:
    return sum(1 for event in game.get("events", []) if event.get("type") == "wolf_team_plan_fallback")


def _wolf_plan_attempt_count(game: dict[str, Any]) -> int:
    plan_count = sum(1 for event in game.get("events", []) if event.get("type") == "wolf_team_plan")
    return max(plan_count, _wolf_plan_fallback_count(game))


def compute_wolf_plan_outcome_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    """按稳定决策键汇总每次计划的唯一终态结果。"""
    decisions: dict[tuple[Any, ...], dict[str, Any]] = {}
    for game_index, game in enumerate(games):
        game_id = game.get("game_id") or f"game-index:{game_index}"
        pending_key: tuple[Any, ...] | None = None
        pending_reason: Any = None
        for event_index, event in enumerate(game.get("events", [])):
            event_type = event.get("type")
            if event_type not in {"wolf_team_plan", "wolf_team_plan_fallback"}:
                continue
            payload = event.get("payload") or {}
            decision_id = payload.get("decision_id") or payload.get("trace_id")
            night_number = payload.get("night_number")
            if decision_id is not None:
                key = ("decision", game_id, str(decision_id))
            elif night_number is not None:
                key = ("legacy-night", game_id, night_number)
            elif event_type == "wolf_team_plan" and pending_key is not None:
                key = pending_key
            elif (
                event_type == "wolf_team_plan_fallback"
                and pending_key is not None
                and pending_reason == payload.get("reason")
            ):
                key = pending_key
            else:
                key = ("legacy-event", game_id, event_index)

            decision = decisions.setdefault(
                key,
                {
                    "plan": None,
                    "fallback_present": False,
                    "fallback_reason": None,
                    "normalization_triggered": False,
                    "normalization_repairs": [],
                },
            )
            repairs = payload.get("normalization_repairs")
            if isinstance(repairs, list):
                decision["normalization_repairs"] = list(dict.fromkeys([
                    *decision["normalization_repairs"],
                    *(str(item) for item in repairs if item),
                ]))
            decision["normalization_triggered"] = bool(
                decision["normalization_triggered"]
                or payload.get("normalization_triggered") is True
                or decision["normalization_repairs"]
            )
            if event_type == "wolf_team_plan_fallback":
                decision["fallback_present"] = True
                decision["fallback_reason"] = payload.get("reason")
                pending_key = key
                pending_reason = payload.get("reason")
            else:
                decision["plan"] = payload
                if pending_key == key:
                    pending_key = None
                    pending_reason = None

    strategy_reasons = {
        "strategy_validation_failed",
        "weak_plan",
        "weak_plan_evidence",
        "weak_evidence",
        "quorum_not_met",
        "insufficient_quorum",
        "consensus_not_reached",
        "no_valid_supporters",
    }
    outcomes: list[str] = []
    for decision in decisions.values():
        reason = decision["fallback_reason"]
        plan = decision["plan"] or {}
        if reason == "schema_validation_failed":
            outcomes.append("schema_terminal_fallback")
        elif reason in strategy_reasons:
            outcomes.append("strategy_terminal_fallback")
        elif decision["fallback_present"] or plan.get("consensus_method") == "fallback":
            outcomes.append("other_terminal_fallback")
        elif decision["normalization_triggered"]:
            outcomes.append("normalization_success")
        else:
            outcomes.append("llm_success")

    total = len(decisions)
    normalization_success = outcomes.count("normalization_success")
    llm_success = outcomes.count("llm_success")
    schema_fallback = outcomes.count("schema_terminal_fallback")
    strategy_fallback = outcomes.count("strategy_terminal_fallback")
    other_fallback = outcomes.count("other_terminal_fallback")
    terminal_fallback = schema_fallback + strategy_fallback + other_fallback
    normalization_triggered = sum(
        bool(decision["normalization_triggered"])
        for decision in decisions.values()
    )
    denominator = total or None
    normalization_denominator = normalization_triggered or None
    return {
        "wolf_team_plan_outcome_metrics_supported": denominator is not None,
        "wolf_team_plan_total_count": total,
        "wolf_team_plan_normalization_metrics_supported": (
            normalization_denominator is not None
        ),
        "wolf_team_plan_normalization_triggered_count": normalization_triggered,
        "wolf_team_plan_normalization_success_count": normalization_success,
        "wolf_team_plan_llm_success_count": llm_success,
        "wolf_team_plan_schema_terminal_fallback_count": schema_fallback,
        "wolf_team_plan_strategy_terminal_fallback_count": strategy_fallback,
        "wolf_team_plan_other_terminal_fallback_count": other_fallback,
        "wolf_team_plan_terminal_fallback_count": terminal_fallback,
        "wolf_team_plan_normalization_success_rate": (
            normalization_success / normalization_denominator
            if normalization_denominator else None
        ),
        "wolf_team_plan_schema_terminal_fallback_rate": (
            schema_fallback / denominator if denominator else None
        ),
        "wolf_team_plan_strategy_terminal_fallback_rate": (
            strategy_fallback / denominator if denominator else None
        ),
        "wolf_team_plan_other_terminal_fallback_rate": (
            other_fallback / denominator if denominator else None
        ),
        "wolf_team_plan_terminal_fallback_rate": (
            terminal_fallback / denominator if denominator else None
        ),
        "wolf_plan_schema_fallback_rate": (
            schema_fallback / denominator if denominator else None
        ),
        "wolf_plan_strategy_fallback_rate": (
            strategy_fallback / denominator if denominator else None
        ),
    }


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


def _trace_failed(trace: dict[str, Any]) -> bool:
    return any(trace.get(field) for field in _FAILURE_TRACE_FIELDS)


def _roles(game: dict[str, Any]) -> dict[Any, str]:
    players = game.get("players") or {}
    return {
        player_id: str(data.get("role", "")).lower()
        for player_id, data in players.items()
        if isinstance(data, dict)
    }


def _sheriff_werewolf_rate(games: list[dict[str, Any]]) -> float:
    total = 0
    werewolf_sheriffs = 0
    for game in games:
        roles = _roles(game)
        for event in game.get("events", []):
            if event.get("type") != "sheriff_elected":
                continue
            payload = event.get("payload") or {}
            sheriff_id = payload.get("sheriff_id") or payload.get("player_id")
            if not sheriff_id:
                continue
            total += 1
            if roles.get(sheriff_id) == "werewolf":
                werewolf_sheriffs += 1
    return werewolf_sheriffs / total if total else 0.0


def _sheriff_vote_fallback_rate(action_trace_records: list[dict[str, Any]]) -> float:
    sheriff_vote_records = [
        record
        for record in action_trace_records
        if _is_sheriff_vote_task(record.get("task"))
    ]
    if not sheriff_vote_records:
        return 0.0
    failures = sum(1 for record in sheriff_vote_records if _trace_failed(record["trace"]))
    return failures / len(sheriff_vote_records)


def _is_sheriff_vote_task(task: Any) -> bool:
    normalized = str(task or "").lower().replace("-", "_")
    return normalized == "sheriff_vote"


def _hunter_friendly_fire_rate(games: list[dict[str, Any]]) -> float:
    hunter_shots = 0
    friendly_fire = 0
    for game in games:
        roles = _roles(game)
        for death in game.get("deaths", []):
            if death.get("reason") != "hunter_shot":
                continue
            hunter_shots += 1
            if roles.get(death.get("player_id")) != "werewolf":
                friendly_fire += 1
    return friendly_fire / hunter_shots if hunter_shots else 0.0


def _weak_plan_kill_rate(games: list[dict[str, Any]]) -> float:
    planned_kills = 0
    weak_or_missing_plan_kills = 0
    for game in games:
        weak_kills, total_kills = _weak_wolf_plan_kill_counts(game)
        weak_or_missing_plan_kills += weak_kills
        planned_kills += total_kills
    return weak_or_missing_plan_kills / planned_kills if planned_kills else 0.0


def _power_role_fallback_rate(action_trace_records: list[dict[str, Any]]) -> float:
    records = []
    for record in action_trace_records:
        actor = record.get("actor")
        if not actor:
            continue
        role = _roles(record["game"]).get(actor)
        if role in _POWER_ROLES:
            records.append(record)
    if not records:
        return 0.0
    failures = sum(1 for record in records if _trace_failed(record["trace"]))
    return failures / len(records)


def _weak_wolf_plan_kills(game: dict[str, Any]) -> int:
    weak_kills, _total_kills = _weak_wolf_plan_kill_counts(game)
    return weak_kills


def _weak_wolf_plan_kill_counts(game: dict[str, Any]) -> tuple[int, int]:
    plan_quality_by_night: dict[Any, str] = {}
    for event in game.get("events", []):
        if event.get("type") == "wolf_team_plan":
            payload = event.get("payload") or {}
            plan_quality_by_night[payload.get("night_number")] = payload.get("evidence_quality", "none")

    count = 0
    total = 0
    for event in game.get("events", []):
        if event.get("type") != "wolf_kill_selected":
            continue
        payload = event.get("payload") or {}
        if payload.get("reason") != "wolf_team_plan":
            continue
        total += 1
        quality = plan_quality_by_night.get(payload.get("night_number"), "none")
        if quality in ("none", "weak"):
            count += 1
    return count, total


def _fallback_plan_kill_count(game: dict[str, Any]) -> int:
    _missing_evidence, total = _fallback_plan_kill_evidence_counts(game)
    return total


def _fallback_plan_kill_without_target_evidence_count(game: dict[str, Any]) -> int:
    missing_evidence, _total = _fallback_plan_kill_evidence_counts(game)
    return missing_evidence


def _fallback_plan_kill_evidence_counts(game: dict[str, Any]) -> tuple[int, int]:
    fallback_evidence_targets_by_night: dict[Any, set[Any]] = {}
    for event in game.get("events", []):
        if event.get("type") != "wolf_team_plan":
            continue
        payload = event.get("payload") or {}
        if payload.get("consensus_method") != "fallback":
            continue
        night = payload.get("night_number")
        targets = {
            item.get("target")
            for item in payload.get("evidence_from_discussion") or []
            if isinstance(item, dict) and item.get("target")
        }
        fallback_evidence_targets_by_night[night] = targets

    missing_evidence = 0
    total = 0
    for event in game.get("events", []):
        if event.get("type") != "wolf_kill_selected":
            continue
        payload = event.get("payload") or {}
        if payload.get("reason") != "wolf_team_plan":
            continue
        night = payload.get("night_number")
        if night not in fallback_evidence_targets_by_night:
            continue
        total += 1
        if payload.get("target_id") not in fallback_evidence_targets_by_night[night]:
            missing_evidence += 1
    return missing_evidence, total


def _vote_concentration(event: dict[str, Any]) -> float:
    payload = event.get("payload") or {}
    voters = payload.get("voters")
    if isinstance(voters, dict) and voters:
        targets = list(voters.values())
    else:
        votes = payload.get("votes")
        if not isinstance(votes, list) or not votes:
            return 0.0
        targets = [
            vote.get("target")
            for vote in votes
            if isinstance(vote, dict) and vote.get("target")
        ]
    if not targets:
        return 0.0
    counts: dict[str, int] = {}
    for target in targets:
        counts[target] = counts.get(target, 0) + 1
    return max(counts.values()) / len(targets)


def _seer_day1_exile_rate(games: list[dict[str, Any]]) -> float:
    if not games:
        return 0.0
    hits = 0
    for game in games:
        roles = {pid: data.get("role") for pid, data in (game.get("players") or {}).items()}
        for event in game.get("events", []):
            if event.get("type") == "vote_resolved" and (event.get("payload") or {}).get("exiled"):
                payload = event.get("payload") or {}
                day_number = payload.get("day_number")
                if day_number is not None and str(day_number) != "1":
                    continue
                exiled = payload.get("exiled")
                if roles.get(exiled) == "seer":
                    hits += 1
                break
    return hits / len(games)


def _witch_night1_death_rate(games: list[dict[str, Any]]) -> float:
    if not games:
        return 0.0
    hits = 0
    for game in games:
        roles = {pid: data.get("role") for pid, data in (game.get("players") or {}).items()}
        for death in game.get("deaths", []):
            if (
                roles.get(death.get("player_id")) == "witch"
                and death.get("reason") == "wolf_kill"
                and _death_night_number(death) == 1
            ):
                hits += 1
                break
    return hits / len(games)


def _witch_wolf_kill_death_rate(games: list[dict[str, Any]]) -> float:
    if not games:
        return 0.0
    hits = 0
    for game in games:
        roles = {pid: data.get("role") for pid, data in (game.get("players") or {}).items()}
        for death in game.get("deaths", []):
            if roles.get(death.get("player_id")) == "witch" and death.get("reason") == "wolf_kill":
                hits += 1
                break
    return hits / len(games)


def _death_night_number(death: dict[str, Any]) -> int | None:
    batch = str(death.get("resolution_batch") or "")
    match = re.fullmatch(r"night_(\d+)", batch)
    if match:
        return int(match.group(1))
    value = death.get("night_number")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _template_vote_reason_counts(games: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    template = 0
    for game in games:
        for event in game.get("events", []):
            if event.get("type") != "vote_resolved":
                continue
            payload = event.get("payload") or {}
            for vote in payload.get("votes") or []:
                if not isinstance(vote, dict):
                    continue
                reason = str(vote.get("reason") or "")
                if not reason:
                    continue
                total += 1
                if _is_template_vote_reason(reason):
                    template += 1
    return template, total


def _is_template_vote_reason(reason: str) -> bool:
    return all(marker in reason for marker in _TEMPLATE_VOTE_REASON_MARKERS)
