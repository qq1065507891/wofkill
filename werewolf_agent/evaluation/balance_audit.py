"""Batch balance audit helpers for saved game logs.

The functions here are intentionally pure: they consume saved JSON-style game
dicts and never call model providers or mutate game state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

_FAILURE_TRACE_FIELDS = ("fallback_reason", "parse_error", "structured_failure_reason")
_POWER_ROLES = {"seer", "witch", "hunter", "idiot"}


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

    weak_wolf_plan_kill_count = sum(_weak_wolf_plan_kills(game) for game in games)
    vote_concentrations = [_vote_concentration(event) for game in games for event in game.get("events", []) if event.get("type") == "vote_resolved"]
    warnings: list[str] = []

    wolf_win_rate = wolf_wins / game_count if game_count else 0.0
    good_win_rate = good_wins / game_count if game_count else 0.0
    fallback_action_rate = fallback_count / len(action_traces) if action_traces else 0.0
    schema_failure_rate = schema_failures / len(action_traces) if action_traces else 0.0
    seer_day1_exile_rate = _seer_day1_exile_rate(games)
    witch_night1_death_rate = _witch_night1_death_rate(games)
    sheriff_werewolf_rate = _sheriff_werewolf_rate(games)
    sheriff_vote_fallback_rate = _sheriff_vote_fallback_rate(action_trace_records)
    hunter_friendly_fire_rate = _hunter_friendly_fire_rate(games)
    weak_plan_kill_rate = _weak_plan_kill_rate(games)
    power_role_fallback_rate = _power_role_fallback_rate(action_trace_records)
    mean_vote_concentration = (
        sum(vote_concentrations) / len(vote_concentrations)
        if vote_concentrations else 0.0
    )

    if game_count >= 10 and wolf_win_rate > 0.75:
        warnings.append("wolf_win_rate_high")
    if schema_failure_rate > 0.05:
        warnings.append("schema_failure_high")
    if game_count >= 10 and seer_day1_exile_rate > 0.35:
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

    return {
        "games": game_count,
        "wolf_win_rate": wolf_win_rate,
        "good_win_rate": good_win_rate,
        "fallback_action_rate": fallback_action_rate,
        "schema_failure_rate": schema_failure_rate,
        "seer_day1_exile_rate": seer_day1_exile_rate,
        "d1_seer_exile_rate": seer_day1_exile_rate,
        "witch_night1_death_rate": witch_night1_death_rate,
        "sheriff_werewolf_rate": sheriff_werewolf_rate,
        "sheriff_vote_fallback_rate": sheriff_vote_fallback_rate,
        "hunter_friendly_fire_rate": hunter_friendly_fire_rate,
        "weak_plan_kill_rate": weak_plan_kill_rate,
        "power_role_fallback_rate": power_role_fallback_rate,
        "mean_vote_concentration": mean_vote_concentration,
        "weak_wolf_plan_kill_count": weak_wolf_plan_kill_count,
        "warnings": warnings,
    }


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
                            "game": game,
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
            if roles.get(death.get("player_id")) == "witch" and death.get("reason") == "wolf_kill":
                hits += 1
                break
    return hits / len(games)
