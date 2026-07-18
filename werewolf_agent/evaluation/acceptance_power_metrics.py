# -*- coding: utf-8 -*-
"""
投影猎人和女巫伤害决策的证据完整性验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    projection_support,
)


_POWER_CHAIN_EVENT_TYPES = frozenset({
    "hunter_shot_opportunity",
    "hunter_shot_selected",
    "hunter_shot_declined",
    "hunter_shot_blocked",
    "hunter_shot_resolved",
    "seer_check_opportunity",
    "seer_check_selected",
    "seer_check_repaired",
    "seer_check_skipped",
    "seer_check_resolved",
})


def compute_power_acceptance_metrics(
    games: Iterable[Any],
) -> dict[str, Any]:
    """对账伤害事实与动作轨迹并投影神职证据指标。"""
    return _compute_power_acceptance_metrics_from_normalized(
        normalize_acceptance_games(games)
    )


def _compute_power_acceptance_metrics_from_normalized(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """消费同一调用链刚完成验证的不可变游戏快照。"""
    projection_is_supported, projection_reason = projection_support(games)
    power_decisions: list[dict[str, Any]] = []
    for game in games:
        power_trace_candidates: list[tuple[str, str | None, str, dict[str, Any]]] = []
        power_damage_events: list[tuple[str, str | None, str]] = []
        for event in game.get("events", []):
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "player_died" and payload.get("reason") in {
                "hunter_shot", "witch_poison",
            }:
                power_damage_events.append((
                    str(payload.get("reason")),
                    str(payload.get("source_player_id"))
                    if payload.get("source_player_id") else None,
                    str(payload.get("player_id") or ""),
                ))
            if event_type != "action_trace_audit":
                continue
            trace = payload.get("action_trace")
            if not isinstance(trace, Mapping):
                continue
            if trace.get("final_action_type") in {"hunter_shot", "use_poison"}:
                evidence = trace.get("power_role_evidence")
                power_trace_candidates.append((
                    str(trace.get("final_action_type")),
                    str(payload.get("player_id")) if payload.get("player_id") else None,
                    str((evidence or {}).get("target_id") or "")
                    if isinstance(evidence, Mapping) else "",
                    evidence if isinstance(evidence, Mapping) else {},
                ))

        # deaths 是权威结果；player_died 必须与之逐项一致，不能互相掩盖遗漏。
        authoritative_deaths = [
            (
                str(death["reason"]),
                str(death.get("source_player_id"))
                if death.get("source_player_id") else None,
                str(death.get("player_id") or ""),
            )
            for death in game.get("deaths", [])
            if isinstance(death, Mapping) and death.get("reason") in {
                "hunter_shot", "witch_poison",
            }
        ]
        death_counts = Counter(authoritative_deaths)
        event_counts = Counter(power_damage_events)
        damage_sources_match = death_counts == event_counts
        reconciled_damage_events = list((death_counts | event_counts).elements())
        used_trace_indexes: set[int] = set()
        for reason, source, target in reconciled_damage_events:
            if not damage_sources_match:
                power_decisions.append({})
                continue
            expected_action = "hunter_shot" if reason == "hunter_shot" else "use_poison"
            match_index = next((
                index
                for index, (action, actor, trace_target, _evidence)
                in enumerate(power_trace_candidates)
                if index not in used_trace_indexes
                and action == expected_action
                and trace_target == target
                and (reason != "hunter_shot" or actor == source)
            ), None)
            if match_index is None:
                power_decisions.append({})
            else:
                used_trace_indexes.add(match_index)
                power_decisions.append(power_trace_candidates[match_index][3])

    complete_power_decisions = sum(
        _power_role_evidence_complete(evidence) for evidence in power_decisions
    )
    power_count = len(power_decisions)
    opportunity_metrics = _power_role_opportunity_metrics(
        games,
        projection_is_supported=projection_is_supported,
        projection_reason=projection_reason,
    )
    return {
        "power_role_evidence_metrics_supported": (
            projection_is_supported and power_count > 0
        ),
        "power_role_evidence_metrics_unsupported_reason": (
            projection_reason if not projection_is_supported
            else None if power_count else "no_power_role_damage_decisions"
        ),
        "power_role_damage_decision_count": power_count,
        "power_role_evidence_complete_count": complete_power_decisions,
        "power_role_evidence_completeness_rate": (
            complete_power_decisions / power_count
            if projection_is_supported and power_count else None
        ),
        **opportunity_metrics,
    }


def _power_role_opportunity_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    projection_is_supported: bool,
    projection_reason: str | None,
) -> dict[str, Any]:
    """按神职实际机会而非伤害结果计算技能选择分母。"""
    if not projection_is_supported:
        return _unsupported_power_opportunity_metrics(projection_reason)
    if any(game.get("status") != "finished" for game in games):
        return _unsupported_power_opportunity_metrics("unfinished_game")
    valid, reason, counts = _validated_power_opportunity_counts(games)
    if not valid:
        return _unsupported_power_opportunity_metrics(reason)
    opportunities = counts["opportunity"]
    return {
        "power_role_opportunity_metrics_supported": True,
        "power_role_opportunity_metrics_unsupported_reason": None,
        "power_role_opportunity_count": opportunities,
        "power_role_selected_count": counts["selected"],
        "power_role_repaired_count": counts["repaired"],
        "power_role_declined_count": counts["declined"],
        "power_role_skipped_count": counts["skipped"],
        "power_role_blocked_count": counts["blocked"],
        "power_role_resolved_count": counts["resolved"],
        "power_role_selection_rate": (
            counts["selected"] / opportunities if opportunities else None
        ),
    }

def _unsupported_power_opportunity_metrics(reason: str | None) -> dict[str, Any]:
    return {
        "power_role_opportunity_metrics_supported": False,
        "power_role_opportunity_metrics_unsupported_reason": (
            reason or "unsupported_power_opportunity_metrics"
        ),
        "power_role_opportunity_count": None,
        "power_role_selected_count": None,
        "power_role_repaired_count": None,
        "power_role_declined_count": None,
        "power_role_skipped_count": None,
        "power_role_blocked_count": None,
        "power_role_resolved_count": None,
        "power_role_selection_rate": None,
    }


def _validated_power_opportunity_counts(
    games: Sequence[Mapping[str, Any]],
) -> tuple[bool, str | None, Counter[str]]:
    counts: Counter[str] = Counter()
    for game in games:
        game_id = game.get("game_id")
        events = game.get("events")
        players = game.get("players")
        if not isinstance(game_id, str) or not isinstance(events, (list, tuple)) or not isinstance(players, Mapping):
            return False, "invalid_power_opportunity_log", Counter()
        ordered, order_reason = _power_chain_log_is_ordered(events, game_id)
        if not ordered:
            return False, order_reason, Counter()
        used_event_ids: set[str] = set()
        consumed_indexes: set[int] = set()
        identities: set[tuple[str, str, int]] = set()
        for index, opportunity in enumerate(events):
            if not isinstance(opportunity, Mapping):
                return False, "invalid_power_opportunity_event", Counter()
            spec = _power_opportunity_spec(str(opportunity.get("type") or ""))
            if spec is None:
                continue
            role, choice_types, resolution_type, resolution_visibility = spec
            payload = opportunity.get("payload")
            if not isinstance(payload, Mapping) or not _is_canonical_power_event(
                opportunity, game_id, "moderator_only", used_event_ids
            ):
                return False, "noncanonical_power_opportunity", Counter()
            actor_id = payload.get("actor_id")
            night_number = payload.get("night_number")
            player = players.get(actor_id) if isinstance(actor_id, str) else None
            if (
                not isinstance(player, Mapping)
                or player.get("role") != role
                or isinstance(night_number, bool)
                or not isinstance(night_number, int)
                or night_number < 1
            ):
                return False, "invalid_power_opportunity_actor", Counter()
            identity = (role, actor_id, night_number)
            if identity in identities:
                return False, "duplicate_power_opportunity_identity", Counter()
            identities.add(identity)
            consumed_indexes.add(index)
            choice_index = _find_power_chain_event(
                events, index + 1, actor_id, night_number, choice_types,
                "moderator_only", game_id, used_event_ids,
            )
            if choice_index is None:
                return False, "missing_power_opportunity_choice", Counter()
            consumed_indexes.add(choice_index)
            resolution_index = _find_power_chain_event(
                events, choice_index + 1, actor_id, night_number, {resolution_type},
                resolution_visibility, game_id, used_event_ids,
                allow_missing_night=(role == "hunter"),
            )
            if resolution_index is None:
                return False, "missing_power_opportunity_resolution", Counter()
            consumed_indexes.add(resolution_index)
            counts["opportunity"] += 1
            choice_type = str(events[choice_index].get("type"))
            counts[_power_choice_bucket(choice_type)] += 1
            counts["resolved"] += 1
        if any(
            isinstance(event, Mapping)
            and str(event.get("type") or "") in _POWER_CHAIN_EVENT_TYPES
            and index not in consumed_indexes
            for index, event in enumerate(events)
        ):
            return False, "unconsumed_power_chain_event", Counter()
    return True, None, counts


def _power_chain_log_is_ordered(
    events: Sequence[Any],
    game_id: str,
) -> tuple[bool, str | None]:
    """检查相关 V2 日志在事件列表中具备唯一且严格递增的序号。"""
    seen_event_ids: set[str] = set()
    last_sequence = -1
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type") or "") not in _POWER_CHAIN_EVENT_TYPES:
            continue
        event_id = event.get("event_id")
        sequence_number = event.get("sequence_number")
        if (
            event.get("schema_version") != "2"
            or event.get("game_id") != game_id
            or not isinstance(event_id, str)
            or isinstance(sequence_number, bool)
            or not isinstance(sequence_number, int)
            or sequence_number < 0
            or event_id != f"{game_id}:e{sequence_number:06d}"
            or not event.get("occurred_at")
            or event_id in seen_event_ids
            or sequence_number <= last_sequence
        ):
            return False, "out_of_order_power_chain_log"
        seen_event_ids.add(event_id)
        last_sequence = sequence_number
    return True, None


def _power_opportunity_spec(
    event_type: str,
) -> tuple[str, set[str], str, str] | None:
    if event_type == "hunter_shot_opportunity":
        return (
            "hunter",
            {"hunter_shot_selected", "hunter_shot_declined", "hunter_shot_blocked"},
            "hunter_shot_resolved",
            "public",
        )
    if event_type == "seer_check_opportunity":
        return (
            "seer",
            {"seer_check_selected", "seer_check_repaired", "seer_check_skipped"},
            "seer_check_resolved",
            "moderator_only",
        )
    return None


def _power_choice_bucket(event_type: str) -> str:
    return {
        "hunter_shot_selected": "selected",
        "seer_check_selected": "selected",
        "seer_check_repaired": "repaired",
        "hunter_shot_declined": "declined",
        "seer_check_skipped": "skipped",
        "hunter_shot_blocked": "blocked",
    }[event_type]


def _is_canonical_power_event(
    event: Mapping[str, Any],
    game_id: str,
    visibility: str,
    used_event_ids: set[str],
) -> bool:
    event_id = event.get("event_id")
    sequence_number = event.get("sequence_number")
    if (
        event.get("visibility") != visibility
        or event.get("schema_version") != "2"
        or event.get("game_id") != game_id
        or not isinstance(event_id, str)
        or isinstance(sequence_number, bool)
        or not isinstance(sequence_number, int)
        or sequence_number < 0
        or event_id != f"{game_id}:e{sequence_number:06d}"
        or not event.get("occurred_at")
        or event_id in used_event_ids
    ):
        return False
    used_event_ids.add(event_id)
    return True


def _find_power_chain_event(
    events: Sequence[Any],
    start: int,
    actor_id: str,
    night_number: int,
    allowed_types: set[str],
    visibility: str,
    game_id: str,
    used_event_ids: set[str],
    *,
    allow_missing_night: bool = False,
) -> int | None:
    for index in range(start, len(events)):
        event = events[index]
        if not isinstance(event, Mapping) or str(event.get("type") or "") not in allowed_types:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("actor_id") != actor_id:
            continue
        if payload.get("night_number") not in ({None, night_number} if allow_missing_night else {night_number}):
            continue
        if _is_canonical_power_event(event, game_id, visibility, used_event_ids):
            return index
        return None
    return None


def _power_role_evidence_complete(evidence: dict[str, Any]) -> bool:
    comparison = evidence.get("alternative_comparison")
    risk = evidence.get("friendly_fire_risk")
    retain = evidence.get("retain_option")
    if not isinstance(evidence.get("target_id"), str) or not evidence["target_id"]:
        return False
    if not _target_evidence_complete(evidence):
        return False
    if not _friendly_fire_risk_complete(risk):
        return False
    if not _retain_option_complete(retain):
        return False
    if not isinstance(comparison, Mapping):
        return False
    alternatives = comparison.get("legal_alternatives")
    if not isinstance(alternatives, (list, tuple)) or not all(
        isinstance(item, str) and bool(item) for item in alternatives
    ):
        return False
    if not isinstance(comparison.get("no_legal_alternative"), bool):
        return False
    if "alternative_target" not in comparison:
        return False
    alternative_target = comparison["alternative_target"]
    if alternative_target == evidence["target_id"]:
        return False
    if comparison.get("no_legal_alternative") is True:
        return alternative_target is None
    return alternative_target is not None and alternative_target in alternatives


def _target_evidence_complete(evidence: dict[str, Any]) -> bool:
    selected = evidence.get("target_evidence")
    comparison = evidence.get("target_comparison")
    alternatives = evidence.get("alternative_comparison")
    if not isinstance(selected, Mapping) or not isinstance(comparison, Mapping):
        return False
    if not isinstance(alternatives, Mapping):
        return False
    selected_score = selected.get("selected_score")
    selected_signals = selected.get("selected_signals")
    if not _score(selected_score) or not _signals(selected_signals):
        return False
    if (
        comparison.get("selected_score") != selected_score
        or comparison.get("selected_signals") != selected_signals
        or comparison.get("alternative_target")
        != alternatives.get("alternative_target")
        or not isinstance(comparison.get("comparison_basis"), str)
        or not comparison["comparison_basis"].strip()
    ):
        return False
    alternative_target = comparison.get("alternative_target")
    alternative_score = comparison.get("alternative_score")
    alternative_signals = comparison.get("alternative_signals")
    if alternative_target is None:
        return alternative_score is None and _signals(alternative_signals)
    return _score(alternative_score) and _signals(alternative_signals)


def _score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _signals(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, str) and bool(item) for item in value
    )


def _friendly_fire_risk_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    targets = value.get("targets")
    return (
        value.get("status") == "assessed"
        and isinstance(value.get("basis"), str)
        and bool(value["basis"].strip())
        and isinstance(targets, (list, tuple)) and all(
            isinstance(item, str) and bool(item) for item in targets
        )
    )


def _retain_option_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("action") == "no_action"
        and isinstance(value.get("available"), bool)
        and isinstance(value.get("required"), bool)
        and isinstance(value.get("reason"), str)
        and bool(value["reason"].strip())
    )
