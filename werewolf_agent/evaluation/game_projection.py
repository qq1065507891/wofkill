# -*- coding: utf-8 -*-
"""
把运行时状态或历史 JSON 归一化为验收指标使用的完整、不可变游戏投影。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-16

使用示例:
    >>> projection = project_acceptance_game({"game_id": "g1", "events": []})
    >>> projection.status
    'running'
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from werewolf_agent.core.resolution_batches import serialize_resolution_batch
from werewolf_agent.runtime.event_metadata import serialize_game_event


GameStatus = Literal["running", "finished", "aborted"]
_PLAYER_FIELDS = ("id", "role", "alive", "faction")
_METADATA_FIELDS = (
    "phase", "day_number", "night_number", "__source_path",
    "hybrid_master_id", "hybrid_master_faction", "hybrid_result",
)


class _ProjectionValueError(ValueError):
    """标记投影中不支持的非 JSON 值。"""


@dataclass(frozen=True)
class AcceptanceGameProjection:
    """验收计算所需的 moderator-only 单局事实快照。"""

    game_id: str
    events: tuple[Mapping[str, Any], ...]
    players: Mapping[str, Mapping[str, Any]]
    winning_faction: str | None
    status: GameStatus
    termination_reason: str | None = None
    deaths: tuple[Mapping[str, Any], ...] = ()
    steps: int = 0
    supported: bool = True
    unsupported_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """即便调用者直接构造，也递归冻结所有容器。"""
        object.__setattr__(self, "events", tuple(_freeze_json(event) for event in self.events))
        object.__setattr__(self, "players", _freeze_json(_project_players(self.players)))
        object.__setattr__(self, "deaths", tuple(_freeze_json(death) for death in self.deaths))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_mapping(self) -> dict[str, Any]:
        """返回深度可变且严格 JSON-safe 的兼容映射。"""
        return {
            "game_id": self.game_id,
            "events": _thaw_json(self.events),
            "players": _thaw_json(self.players),
            "winning_faction": self.winning_faction,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "deaths": _thaw_json(self.deaths),
            "steps": self.steps,
            "_acceptance_projection_supported": self.supported,
            "_acceptance_projection_unsupported_reason": self.unsupported_reason,
            **_thaw_json(self.metadata),
        }


def project_acceptance_game(
    source: AcceptanceGameProjection | Mapping[str, Any] | Any,
    *,
    steps: int | None = None,
) -> AcceptanceGameProjection:
    """归一化强类型状态或 V1/V2 JSON；非法结构稳定地 fail closed。"""
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
    if "speech_non_empty_rate" not in normalized and "speech_fill_rate" in normalized:
        normalized["speech_non_empty_rate"] = normalized["speech_fill_rate"]
    return normalized


def _from_state(source: Any, *, steps: int | None) -> AcceptanceGameProjection:
    raw_players = getattr(source, "players", {})
    players = {
        str(player_id): {
            "id": str(getattr(player, "id", player_id)),
            "role": str(getattr(player, "role", "")),
            "alive": bool(getattr(player, "alive", False)),
            "faction": _json_scalar(getattr(player, "faction", None)),
        }
        for player_id, player in raw_players.items()
    }
    structural_reason: str | None = None
    try:
        events = tuple(_normalize_event(serialize_game_event(event)) for event in getattr(source, "events", ()))
    except _ProjectionValueError as exc:
        events = ()
        structural_reason = str(exc)
    try:
        deaths = tuple(_normalize_death({
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
        }) for death in getattr(source, "deaths", ()))
    except _ProjectionValueError as exc:
        deaths = ()
        structural_reason = structural_reason or str(exc)
    winner = getattr(source, "winning_faction", None)
    status = _derive_status(getattr(source, "status", "running"), winner, getattr(source, "phase", None))
    reason = structural_reason or _unsupported_reason(
        str(getattr(source, "game_id", "")), players, status=status, winner=winner,
    )
    metadata = {
        "phase": getattr(source, "phase", None),
        "day_number": getattr(source, "day_number", 0),
        "night_number": getattr(source, "night_number", 0),
        **_hybrid_metadata(source, events),
    }
    return AcceptanceGameProjection(
        game_id=str(getattr(source, "game_id", "")), events=events, players=players,
        winning_faction=_optional_string(winner), status=status,
        termination_reason=_optional_string(getattr(source, "termination_reason", None)),
        deaths=deaths, steps=int(steps if steps is not None else getattr(source, "steps", 0) or 0),
        supported=reason is None, unsupported_reason=reason,
        metadata=_normalize_json(metadata),
    )


def _from_mapping(source: Mapping[str, Any], *, steps: int | None) -> AcceptanceGameProjection:
    game_id = str(source.get("game_id") or "")
    raw_events = source.get("events")
    structural_reason = (
        str(source.get("_acceptance_projection_unsupported_reason") or "unsupported_projection")
        if source.get("_acceptance_projection_supported") is False
        else None
    )
    events: tuple[dict[str, Any], ...] = ()
    if "events" not in source:
        structural_reason = "missing_events"
    elif not isinstance(raw_events, (list, tuple)):
        structural_reason = "invalid_events_container"
    else:
        try:
            events = tuple(_normalize_event(event) for event in raw_events)
        except _ProjectionValueError as exc:
            structural_reason = str(exc)

    raw_players = source.get("players")
    players: dict[str, dict[str, Any]] = {}
    if isinstance(raw_players, Mapping):
        for player_id, player in raw_players.items():
            if not isinstance(player, Mapping):
                structural_reason = structural_reason or "invalid_player_entry"
                break
            players[str(player_id)] = {
                key: _normalize_json(player[key])
                for key in _PLAYER_FIELDS if key in player
            }
            players[str(player_id)].setdefault("id", str(player_id))

    raw_deaths = source.get("deaths", ())
    deaths: tuple[dict[str, Any], ...] = ()
    if not isinstance(raw_deaths, (list, tuple)):
        structural_reason = structural_reason or "invalid_deaths_container"
    else:
        try:
            deaths = tuple(_normalize_death(death) for death in raw_deaths)
        except _ProjectionValueError as exc:
            structural_reason = structural_reason or str(exc)

    winner = source.get("winning_faction") or source.get("winner")
    status = _derive_status(source.get("status"), winner, source.get("phase"))
    reason = structural_reason or _unsupported_reason(game_id, players, status=status, winner=winner)
    try:
        metadata = _normalize_json({key: source[key] for key in _METADATA_FIELDS if key in source})
    except _ProjectionValueError:
        metadata = {}
        reason = reason or "invalid_metadata_value"
    return AcceptanceGameProjection(
        game_id=game_id, events=events, players=players,
        winning_faction=_optional_string(winner), status=status,
        termination_reason=_optional_string(source.get("termination_reason")),
        deaths=deaths, steps=int(steps if steps is not None else source.get("steps") or 0),
        supported=reason is None, unsupported_reason=reason, metadata=metadata,
    )


def _normalize_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _ProjectionValueError("invalid_event_entry")
    event_type = value.get("type")
    payload = value.get("payload", {})
    if not isinstance(event_type, str) or not event_type or not isinstance(payload, Mapping):
        raise _ProjectionValueError("invalid_event_entry")
    try:
        return _normalize_json(dict(value))
    except _ProjectionValueError as exc:
        raise _ProjectionValueError("invalid_event_payload") from exc


def _project_players(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """仅保留验收投影需要的玩家公开字段。"""
    projected: dict[str, dict[str, Any]] = {}
    for player_id, player in value.items():
        if not isinstance(player, Mapping):
            raise _ProjectionValueError("invalid_player_entry")
        projected[str(player_id)] = {
            key: player[key] for key in _PLAYER_FIELDS if key in player
        }
        projected[str(player_id)].setdefault("id", str(player_id))
    return projected


def _normalize_death(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _ProjectionValueError("invalid_death_entry")
    try:
        return _normalize_json(dict(value))
    except _ProjectionValueError as exc:
        raise _ProjectionValueError("invalid_death_payload") from exc


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise _ProjectionValueError("non_finite_number")
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_json(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _ProjectionValueError("non_string_mapping_key")
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise _ProjectionValueError(f"unsupported_json_type:{type(value).__name__}")


def _json_scalar(value: Any) -> Any:
    return _normalize_json(value)


def _freeze_json(value: Any) -> Any:
    normalized = _normalize_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _derive_status(raw: Any, winner: Any, phase: Any) -> GameStatus:
    if raw == "aborted":
        return "aborted"
    if raw == "finished" or winner or phase == "finished":
        return "finished"
    return "running"


def _hybrid_metadata(source: Any, events: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    fields = {
        "hybrid_master_id": getattr(source, "hybrid_master_id", None),
        "hybrid_master_faction": getattr(source, "hybrid_master_faction", None),
        "hybrid_result": getattr(source, "hybrid_result", None),
    }
    for event in reversed(events):
        if event.get("type") == "victory":
            payload = event.get("payload") or {}
            for key in fields:
                if fields[key] is None:
                    fields[key] = payload.get(key)
            break
    return fields


def _unsupported_reason(
    game_id: str, players: Mapping[str, Mapping[str, Any]], *, status: GameStatus, winner: Any,
) -> str | None:
    if not game_id:
        return "missing_game_id"
    if not players:
        return "missing_players"
    if any(not player.get("role") for player in players.values()):
        return "missing_player_roles"
    if status == "finished" and not winner:
        return "finished_without_winner"
    return None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


__all__ = [
    "AcceptanceGameProjection", "normalize_acceptance_games", "normalize_quality_score",
    "project_acceptance_game", "projection_support",
]
