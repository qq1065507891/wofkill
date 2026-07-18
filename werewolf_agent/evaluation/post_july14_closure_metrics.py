# -*- coding: utf-8 -*-
"""
从已保存的单局日志聚合 7 月 14 日后修复项的闭环证据与支持性。

作者: Project contributors
创建日期: 2026-07-18

使用示例:
    >>> compute_post_july14_closure_metrics([], quality_recomputer=None)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from werewolf_agent.core.resolution_batches import valid_resolution_batch
from werewolf_agent.evaluation.game_projection import normalize_quality_score
from werewolf_agent.runtime.exposure_audit import (
    is_safe_public_skill_resolution_payload,
)
from werewolf_agent.runtime.wolf_consensus_evidence import (
    ConsensusInvariantViolation,
    WolfTargetStance,
    derive_wolf_consensus_evidence,
)


QualityRecomputer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
_TERMINAL_STATUSES = frozenset({"finished", "aborted"})
_WINNERS = frozenset({"good", "werewolf"})
_PUBLIC_SKILL_RESOLUTIONS = frozenset({
    "hunter_shot_resolved",
    "self_destruct_resolved",
})


def compute_post_july14_closure_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    quality_recomputer: QualityRecomputer | None,
) -> dict[str, Any]:
    """聚合硬门禁所需事实；缺失结构化证据时显式标记 unsupported。"""
    batch_supported, malformed_batches = _resolution_batch_metrics(games)
    v2_supported = _v2_event_logs_supported(games)
    route_supported, same_route_fallbacks = _provider_fallback_route_metrics(
        games,
        v2_supported=v2_supported,
    )
    quality_supported, quality_diffs = _saved_offline_quality_metrics(
        games,
        quality_recomputer=quality_recomputer,
    )
    terminal_supported, finished_without_winner = _terminal_integrity_metrics(games)
    abort_supported, aborted_count, covered_aborts = _abort_terminal_metrics(
        games,
        terminal_supported=terminal_supported,
    )
    consensus = _wolf_consensus_execution_metrics(games)
    reflection_supported, empty_reflection_successes = _reflection_metrics(
        games,
        v2_supported=v2_supported,
    )
    source_supported, source_total, source_traced = _source_trace_metrics(games)
    exposure_supported, public_leaks = _public_exposure_metrics(
        games,
        v2_supported=v2_supported,
    )
    return {
        "resolution_batch_integrity_metrics_supported": batch_supported,
        "malformed_resolution_batch_count": (
            malformed_batches if batch_supported else None
        ),
        "provider_fallback_route_metrics_supported": route_supported,
        "same_route_provider_fallback_count": (
            same_route_fallbacks if route_supported else None
        ),
        "saved_offline_quality_consistency_metrics_supported": quality_supported,
        "saved_offline_quality_diff_count": quality_diffs if quality_supported else None,
        "terminal_integrity_metrics_supported": terminal_supported,
        "finished_without_winner_count": (
            finished_without_winner if terminal_supported else None
        ),
        "abort_terminal_coverage_metrics_supported": abort_supported,
        "aborted_terminal_game_count": aborted_count if abort_supported else None,
        "covered_abort_terminal_game_count": (
            covered_aborts if abort_supported else None
        ),
        "abort_terminal_coverage_rate": (
            covered_aborts / aborted_count if aborted_count else 1.0
        ) if abort_supported else None,
        **consensus,
        "reflection_transaction_metrics_supported": reflection_supported,
        "empty_reflection_success_count": (
            empty_reflection_successes if reflection_supported else None
        ),
        "source_event_traceability_metrics_supported": source_supported,
        "source_event_id_observed_count": source_total if source_supported else None,
        "source_event_id_traced_count": source_traced if source_supported else None,
        "source_event_id_traceability_rate": (
            source_traced / source_total if source_total else None
        ) if source_supported else None,
        "public_exposure_metrics_supported": exposure_supported,
        "public_skill_resolution_leak_count": (
            public_leaks if exposure_supported else None
        ),
    }


def _resolution_batch_metrics(
    games: Sequence[Mapping[str, Any]],
) -> tuple[bool, int]:
    if not games:
        return False, 0
    malformed = 0
    for game in games:
        deaths = game.get("deaths")
        if not isinstance(deaths, (list, tuple)):
            return False, malformed
        for death in deaths:
            if (
                not isinstance(death, Mapping)
                or not isinstance(death.get("resolution_batch_parse_failed"), bool)
            ):
                return False, malformed
            malformed += death.get("resolution_batch_parse_failed") is True
    return True, malformed


def _v2_event_logs_supported(games: Sequence[Mapping[str, Any]]) -> bool:
    if not games:
        return False
    observed = 0
    for game in games:
        events = game.get("events")
        if not isinstance(events, (list, tuple)):
            return False
        for event in events:
            if (
                not isinstance(event, Mapping)
                or event.get("schema_version") != "2"
                or not isinstance(event.get("event_id"), str)
                or not event.get("event_id")
                or not isinstance(event.get("type"), str)
                or not event.get("type")
            ):
                return False
            observed += 1
    return observed > 0


def _provider_fallback_route_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    v2_supported: bool,
) -> tuple[bool, int]:
    if not v2_supported:
        return False, 0
    same_route_count = 0
    for game in games:
        for event in game.get("events", []):
            payload = event.get("payload") or {}
            trace = payload.get("action_trace")
            if not isinstance(trace, Mapping):
                continue
            attempts = trace.get("execution_attempts")
            declared_fallbacks = trace.get("provider_fallback_count", 0)
            if attempts is None:
                if isinstance(declared_fallbacks, int) and declared_fallbacks > 0:
                    return False, same_route_count
                continue
            if not isinstance(attempts, (list, tuple)):
                return False, same_route_count
            previous: Mapping[str, Any] | None = None
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    return False, same_route_count
                route = attempt.get("route_kind")
                if route == "provider_fallback":
                    if previous is None or any(
                        not isinstance(row.get(field), str) or not row.get(field)
                        for row in (previous, attempt)
                        for field in ("provider", "model")
                    ):
                        return False, same_route_count
                    if (
                        previous.get("provider"), previous.get("model")
                    ) == (attempt.get("provider"), attempt.get("model")):
                        same_route_count += 1
                previous = attempt
    return True, same_route_count


def _saved_offline_quality_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    quality_recomputer: QualityRecomputer | None,
) -> tuple[bool, int]:
    if not games or quality_recomputer is None:
        return False, 0
    differences = 0
    for game in games:
        saved = game.get("quality_score")
        if not isinstance(saved, Mapping):
            return False, differences
        try:
            recomputed = quality_recomputer(game)
        except (KeyError, TypeError, ValueError):
            return False, differences
        if not isinstance(recomputed, Mapping):
            return False, differences
        if normalize_quality_score(saved) != normalize_quality_score(recomputed):
            differences += 1
    return True, differences


def _terminal_integrity_metrics(
    games: Sequence[Mapping[str, Any]],
) -> tuple[bool, int]:
    if not games or any(game.get("status") not in _TERMINAL_STATUSES for game in games):
        return False, 0
    missing_winner = sum(
        game.get("status") == "finished"
        and game.get("winning_faction") not in _WINNERS
        for game in games
    )
    return True, missing_winner


def _abort_terminal_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    terminal_supported: bool,
) -> tuple[bool, int, int]:
    if not terminal_supported:
        return False, 0, 0
    aborted = [game for game in games if game.get("status") == "aborted"]
    covered = 0
    for game in aborted:
        events = game.get("events")
        if not isinstance(events, (list, tuple)):
            continue
        terminal_events = [
            event for event in events
            if isinstance(event, Mapping) and event.get("type") == "game_aborted"
        ]
        if len(terminal_events) != 1 or not events or events[-1] is not terminal_events[0]:
            continue
        event = terminal_events[0]
        payload = event.get("payload") or {}
        if (
            isinstance(payload, Mapping)
            and payload.get("termination_reason") == game.get("termination_reason")
            and payload.get("phase") == game.get("phase")
            and event.get("visibility") == "moderator_only"
            and event.get("schema_version") == "2"
            and event.get("game_id") == game.get("game_id")
        ):
            covered += 1
    return True, len(aborted), covered


def _wolf_consensus_execution_metrics(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    opportunities = {"majority": 0, "single_wolf": 0}
    executions = {"majority": 0, "single_wolf": 0}
    supported = True
    observed_stances = 0
    for game in games:
        players = game.get("players")
        events = game.get("events")
        if not isinstance(players, Mapping) or not isinstance(events, (list, tuple)):
            supported = False
            continue
        stances_by_night: dict[int, list[WolfTargetStance]] = defaultdict(list)
        for event in events:
            if not isinstance(event, Mapping) or event.get("type") != "wolf_discussion":
                continue
            payload = event.get("payload") or {}
            raw_stance = payload.get("target_stance") if isinstance(payload, Mapping) else None
            if raw_stance is None:
                continue
            night_number = payload.get("night_number")
            if (
                not isinstance(night_number, int)
                or isinstance(night_number, bool)
                or night_number <= 0
            ):
                supported = False
                continue
            try:
                stance = WolfTargetStance.model_validate(raw_stance)
            except (TypeError, ValueError):
                supported = False
                continue
            stances_by_night[night_number].append(stance)
            observed_stances += 1
        for night_number, stances in stances_by_night.items():
            alive_wolves = _alive_wolf_ids_for_night(
                game,
                night_number=night_number,
            )
            if alive_wolves is None:
                supported = False
                continue
            try:
                consensus = derive_wolf_consensus_evidence(
                    night_number,
                    alive_wolves,
                    stances,
                )
            except (ConsensusInvariantViolation, TypeError, ValueError):
                supported = False
                continue
            status = consensus.primary.status
            if status not in opportunities or consensus.primary.target_id is None:
                continue
            opportunities[status] += 1
            if _matching_wolf_kill_selection(
                events,
                night_number=night_number,
                target_id=consensus.primary.target_id,
                status=status,
            ):
                executions[status] += 1
    supported = supported and observed_stances > 0
    majority_supported = supported and opportunities["majority"] > 0
    single_wolf_supported = supported and opportunities["single_wolf"] > 0
    return {
        "wolf_consensus_execution_metrics_supported": supported,
        "majority_wolf_kill_execution_metrics_supported": majority_supported,
        "majority_wolf_kill_opportunity_count": (
            opportunities["majority"] if majority_supported else None
        ),
        "majority_wolf_kill_execution_count": (
            executions["majority"] if majority_supported else None
        ),
        "majority_wolf_kill_execution_rate": (
            executions["majority"] / opportunities["majority"]
        ) if majority_supported else None,
        "single_wolf_kill_execution_metrics_supported": single_wolf_supported,
        "single_wolf_kill_opportunity_count": (
            opportunities["single_wolf"] if single_wolf_supported else None
        ),
        "single_wolf_kill_execution_count": (
            executions["single_wolf"] if single_wolf_supported else None
        ),
        "single_wolf_kill_execution_rate": (
            executions["single_wolf"] / opportunities["single_wolf"]
        ) if single_wolf_supported else None,
    }


def _alive_wolf_ids_for_night(
    game: Mapping[str, Any],
    *,
    night_number: int,
) -> tuple[str, ...] | None:
    players = game.get("players") or {}
    alive = {
        str(player_id)
        for player_id, player in players.items()
        if isinstance(player, Mapping) and player.get("role") == "werewolf"
    }
    for death in game.get("deaths", []):
        if not isinstance(death, Mapping):
            return None
        player_id = death.get("player_id")
        if player_id not in alive:
            continue
        batch = valid_resolution_batch(
            death.get("resolution_batch") or "",
            parse_failed=bool(death.get("resolution_batch_parse_failed", False)),
        )
        if batch is None:
            return None
        if (
            batch.phase == "day" and batch.number <= night_number
        ) or (
            batch.phase == "night" and batch.number < night_number
        ):
            alive.discard(str(player_id))
    return tuple(sorted(alive)) if alive else None


def _matching_wolf_kill_selection(
    events: Sequence[Mapping[str, Any]],
    *,
    night_number: int,
    target_id: str,
    status: str,
) -> bool:
    marker = f"wolf_kill_selected:stance:{status}:primary"
    return any(
        event.get("type") == "wolf_kill_selected"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("night_number") == night_number
        and event["payload"].get("target_id") == target_id
        and isinstance(event.get("trace_id"), str)
        and marker in event["trace_id"]
        for event in events
        if isinstance(event, Mapping)
    )


def _reflection_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    v2_supported: bool,
) -> tuple[bool, int]:
    if not v2_supported:
        return False, 0
    false_successes = 0
    for game in games:
        if game.get("status") != "finished":
            continue
        reflection_events = [
            event for event in game.get("events", [])
            if event.get("type") == "reflection_complete"
        ]
        if not reflection_events:
            return False, false_successes
        payload = reflection_events[-1].get("payload") or {}
        entries = payload.get("entries") if isinstance(payload, Mapping) else None
        if not isinstance(entries, (list, tuple)):
            return False, false_successes
        if payload.get("status") in {"complete", "partial"} and (
            not entries or payload.get("valid_entry_count") == 0
        ):
            false_successes += 1
    return True, false_successes


def _source_trace_metrics(
    games: Sequence[Mapping[str, Any]],
) -> tuple[bool, int, int]:
    total = 0
    traced = 0
    for game in games:
        for event in game.get("events", []):
            if not isinstance(event, Mapping) or event.get("type") != "wolf_discussion":
                continue
            payload = event.get("payload") or {}
            stance = payload.get("target_stance") if isinstance(payload, Mapping) else None
            if not isinstance(stance, Mapping):
                continue
            total += 1
            if (
                isinstance(event.get("event_id"), str)
                and event.get("event_id")
                and stance.get("source_event_id") == event.get("event_id")
            ):
                traced += 1
    return total > 0, total, traced


def _public_exposure_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    v2_supported: bool,
) -> tuple[bool, int]:
    if not v2_supported:
        return False, 0
    leaks = 0
    for game in games:
        for event in game.get("events", []):
            if event.get("type") not in _PUBLIC_SKILL_RESOLUTIONS:
                continue
            payload = event.get("payload")
            if (
                event.get("visibility") != "public"
                or not isinstance(payload, Mapping)
                or not is_safe_public_skill_resolution_payload(payload)
            ):
                leaks += 1
    return True, leaks


__all__ = ["compute_post_july14_closure_metrics"]
