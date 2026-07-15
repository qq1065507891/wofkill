# -*- coding: utf-8 -*-
"""
把运行时状态或历史 JSON 归一化为验收指标使用的完整游戏投影。

作者: Project contributors
创建日期: 2026-07-15

使用示例:
    >>> projection = project_acceptance_game({"game_id": "g1", "events": []})
    >>> projection.status
    'running'
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Literal, Mapping

from werewolf_agent.core.resolution_batches import serialize_resolution_batch
from werewolf_agent.runtime.event_metadata import serialize_game_event


GameStatus = Literal["running", "finished", "aborted"]


@dataclass(frozen=True)
class AcceptanceGameProjection:
    """验收计算所需的 moderator-only 单局事实快照。"""

    game_id: str
    events: tuple[dict[str, Any], ...]
    players: dict[str, dict[str, Any]]
    winning_faction: str | None
    status: GameStatus
    termination_reason: str | None = None
    deaths: tuple[dict[str, Any], ...] = ()
    steps: int = 0
    supported: bool = True
    unsupported_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """返回供现有验收聚合器只读消费的 JSON-safe 映射。"""
        return {
            "game_id": self.game_id,
            "events": deepcopy(list(self.events)),
            "players": deepcopy(self.players),
            "winning_faction": self.winning_faction,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "deaths": deepcopy(list(self.deaths)),
            "steps": self.steps,
            "_acceptance_projection_supported": self.supported,
            "_acceptance_projection_unsupported_reason": self.unsupported_reason,
            **deepcopy(self.metadata),
        }


def project_acceptance_game(
    source: AcceptanceGameProjection | Mapping[str, Any] | Any,
    *,
    steps: int | None = None,
) -> AcceptanceGameProjection:
    """归一化强类型状态或 V1/V2 JSON；旧日志只推导状态，不补造事实。"""
    if isinstance(source, AcceptanceGameProjection):
        return source
    if isinstance(source, Mapping):
        return _from_mapping(source, steps=steps)
    return _from_state(source, steps=steps)


def normalize_acceptance_games(
    games: list[AcceptanceGameProjection | Mapping[str, Any] | Any],
) -> list[dict[str, Any]]:
    """统一验收入口，并保留每局不支持原因。"""
    return [project_acceptance_game(game).to_mapping() for game in games]


def projection_support(games: list[Mapping[str, Any]]) -> tuple[bool, str | None]:
    """按输入顺序返回首个稳定的不支持原因。"""
    for game in games:
        if game.get("_acceptance_projection_supported") is not True:
            return False, str(
                game.get("_acceptance_projection_unsupported_reason")
                or "incomplete_projection"
            )
    return bool(games), None if games else "no_games"


def normalize_quality_score(quality_score: Mapping[str, Any]) -> dict[str, Any]:
    """读取一周期 V1 speech_fill_rate，并让显式新字段优先。"""
    normalized = dict(quality_score)
    if (
        "speech_non_empty_rate" not in normalized
        and "speech_fill_rate" in normalized
    ):
        normalized["speech_non_empty_rate"] = normalized["speech_fill_rate"]
    return normalized


def _from_state(source: Any, *, steps: int | None) -> AcceptanceGameProjection:
    players = {
        str(player_id): {
            "role": str(getattr(player, "role", "")),
            "alive": bool(getattr(player, "alive", False)),
            "faction": getattr(player, "faction", None),
        }
        for player_id, player in getattr(source, "players", {}).items()
    }
    events = tuple(
        deepcopy(serialize_game_event(event))
        for event in getattr(source, "events", ())
    )
    deaths = tuple(
        {
            "player_id": death.player_id,
            "reason": death.reason,
            "timing": death.timing,
            "resolution_batch": serialize_resolution_batch(death.resolution_batch)[0],
            "resolution_batch_parse_failed": bool(
                death.resolution_batch_parse_failed
                or serialize_resolution_batch(death.resolution_batch)[1]
            ),
            "source_player_id": death.source_player_id,
            "can_leave_last_words": death.can_leave_last_words,
            "triggered_skills": list(death.triggered_skills),
        }
        for death in getattr(source, "deaths", ())
    )
    winner = getattr(source, "winning_faction", None)
    raw_status = getattr(source, "status", "running")
    status = _derive_status(raw_status, winner, getattr(source, "phase", None))
    reason = _unsupported_reason(
        str(getattr(source, "game_id", "")),
        events,
        players,
        status=status,
        winner=winner,
    )
    metadata = {
        "phase": getattr(source, "phase", None),
        "day_number": getattr(source, "day_number", 0),
        "night_number": getattr(source, "night_number", 0),
        **_hybrid_metadata(source, events),
    }
    return AcceptanceGameProjection(
        game_id=str(getattr(source, "game_id", "")),
        events=events,
        players=players,
        winning_faction=winner,
        status=status,
        termination_reason=getattr(source, "termination_reason", None),
        deaths=deaths,
        steps=int(steps if steps is not None else getattr(source, "steps", 0) or 0),
        supported=reason is None,
        unsupported_reason=reason,
        metadata=_json_safe(metadata),
    )


def _from_mapping(source: Mapping[str, Any], *, steps: int | None) -> AcceptanceGameProjection:
    raw_events = source.get("events")
    events = tuple(deepcopy(dict(event)) for event in raw_events if isinstance(event, Mapping)) \
        if isinstance(raw_events, (list, tuple)) else ()
    raw_players = source.get("players")
    players = {
        str(player_id): deepcopy(dict(player))
        for player_id, player in raw_players.items()
        if isinstance(player, Mapping)
    } if isinstance(raw_players, Mapping) else {}
    raw_deaths = source.get("deaths")
    deaths = tuple(deepcopy(dict(death)) for death in raw_deaths if isinstance(death, Mapping)) \
        if isinstance(raw_deaths, (list, tuple)) else ()
    winner = source.get("winning_faction") or source.get("winner")
    status = _derive_status(source.get("status"), winner, source.get("phase"))
    game_id = str(source.get("game_id") or "")
    reason = _unsupported_reason(
        game_id,
        events,
        players,
        status=status,
        winner=winner,
        source=source,
    )
    metadata = _json_safe({
        key: source[key]
        for key in (
            "phase", "day_number", "night_number", "__source_path",
            "hybrid_master_id", "hybrid_master_faction", "hybrid_result",
        )
        if key in source
    })
    return AcceptanceGameProjection(
        game_id=game_id,
        events=events,
        players=players,
        winning_faction=str(winner) if winner else None,
        status=status,
        termination_reason=(
            str(source["termination_reason"])
            if source.get("termination_reason") is not None else None
        ),
        deaths=deaths,
        steps=int(steps if steps is not None else source.get("steps") or 0),
        supported=reason is None,
        unsupported_reason=reason,
        metadata=metadata,
    )


def _derive_status(raw: Any, winner: Any, phase: Any) -> GameStatus:
    if raw == "aborted":
        return "aborted"
    if raw == "finished":
        return "finished"
    if winner or phase == "finished":
        return "finished"
    return "running"


def _hybrid_metadata(
    source: Any,
    events: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    fields = {
        "hybrid_master_id": getattr(source, "hybrid_master_id", None),
        "hybrid_master_faction": getattr(source, "hybrid_master_faction", None),
        "hybrid_result": getattr(source, "hybrid_result", None),
    }
    for event in reversed(events):
        if event.get("type") != "victory":
            continue
        payload = event.get("payload") or {}
        for key in fields:
            if fields[key] is None:
                fields[key] = payload.get(key)
        break
    return fields


def _unsupported_reason(
    game_id: str,
    events: tuple[dict[str, Any], ...],
    players: dict[str, dict[str, Any]],
    *,
    status: GameStatus,
    winner: Any,
    source: Mapping[str, Any] | None = None,
) -> str | None:
    if not game_id:
        return "missing_game_id"
    if source is not None and "events" not in source:
        return "missing_events"
    if not players:
        return "missing_players"
    if any(not player.get("role") for player in players.values()):
        return "missing_player_roles"
    if status == "finished" and not winner:
        return "finished_without_winner"
    return None


def _json_safe(value: Any) -> Any:
    """深拷贝 metadata，并把 Path 等旧日志值收敛为稳定 JSON 标量。"""
    return json.loads(json.dumps(deepcopy(value), ensure_ascii=False, default=str))


__all__ = [
    "AcceptanceGameProjection",
    "normalize_acceptance_games",
    "normalize_quality_score",
    "project_acceptance_game",
    "projection_support",
]
