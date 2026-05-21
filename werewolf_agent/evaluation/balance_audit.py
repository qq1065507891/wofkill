"""Batch balance audit helpers for saved game logs.

The functions here are intentionally pure: they consume saved JSON-style game
dicts and never call model providers or mutate game state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


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

    action_traces = list(_iter_action_traces(games))
    fallback_count = sum(1 for trace in action_traces if trace.get("fallback_reason"))
    schema_failures = sum(
        1 for trace in action_traces
        if "Schema validation error" in str(trace.get("parse_error") or "")
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

    return {
        "games": game_count,
        "wolf_win_rate": wolf_win_rate,
        "good_win_rate": good_win_rate,
        "fallback_action_rate": fallback_action_rate,
        "schema_failure_rate": schema_failure_rate,
        "seer_day1_exile_rate": seer_day1_exile_rate,
        "witch_night1_death_rate": witch_night1_death_rate,
        "mean_vote_concentration": mean_vote_concentration,
        "weak_wolf_plan_kill_count": weak_wolf_plan_kill_count,
        "warnings": warnings,
    }


def _iter_action_traces(games: list[dict[str, Any]]):
    for game in games:
        for event in game.get("events", []):
            payload = event.get("payload") or {}
            trace = payload.get("action_trace")
            if isinstance(trace, dict):
                yield trace
            traces = payload.get("action_traces")
            if isinstance(traces, dict):
                for item in traces.values():
                    if isinstance(item, dict):
                        yield item


def _weak_wolf_plan_kills(game: dict[str, Any]) -> int:
    plan_quality_by_night: dict[Any, str] = {}
    for event in game.get("events", []):
        if event.get("type") == "wolf_team_plan":
            payload = event.get("payload") or {}
            plan_quality_by_night[payload.get("night_number")] = payload.get("evidence_quality", "none")

    count = 0
    for event in game.get("events", []):
        if event.get("type") != "wolf_kill_selected":
            continue
        payload = event.get("payload") or {}
        if payload.get("reason") != "wolf_team_plan":
            continue
        quality = plan_quality_by_night.get(payload.get("night_number"), "none")
        if quality in ("none", "weak"):
            count += 1
    return count


def _vote_concentration(event: dict[str, Any]) -> float:
    payload = event.get("payload") or {}
    voters = payload.get("voters")
    if not isinstance(voters, dict) or not voters:
        return 0.0
    counts: dict[str, int] = {}
    for target in voters.values():
        counts[target] = counts.get(target, 0) + 1
    return max(counts.values()) / len(voters)


def _seer_day1_exile_rate(games: list[dict[str, Any]]) -> float:
    if not games:
        return 0.0
    hits = 0
    for game in games:
        roles = {pid: data.get("role") for pid, data in (game.get("players") or {}).items()}
        for event in game.get("events", []):
            if event.get("type") == "vote_resolved" and (event.get("payload") or {}).get("exiled"):
                exiled = (event.get("payload") or {}).get("exiled")
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
