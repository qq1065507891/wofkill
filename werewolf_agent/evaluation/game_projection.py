# -*- coding: utf-8 -*-
"""
把运行时状态或历史 JSON 归一化为经过边界校验的不可变验收游戏投影。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-24

使用示例:
    >>> projection = project_acceptance_game({"game_id": "g1", "events": []})
    >>> projection.status
    'running'
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

from werewolf_agent.core.resolution_batches import serialize_resolution_batch
from werewolf_agent.evaluation.acceptance_shared import (
    _has_valid_repair_failure_history,
)
from werewolf_agent.runtime.event_metadata import serialize_game_event


GameStatus = Literal["running", "finished", "aborted"]
_GAME_STATUSES = frozenset({"running", "finished", "aborted"})
_PLAYER_FIELDS = ("id", "role", "alive", "faction")
_METADATA_FIELDS = (
    "phase", "day_number", "night_number", "__source_path",
    "hybrid_master_id", "hybrid_master_faction", "hybrid_result",
)
_JSON_BOUND_REASONS = frozenset({
    "cyclic_json_value", "json_depth_exceeded", "json_item_limit_exceeded",
})
_MAX_JSON_DEPTH = 32
_MAX_JSON_ITEMS = 10_000


class _ProjectionValueError(ValueError):
    """标记投影中不支持的结构或非 JSON 值。"""


@dataclass
class _JsonBudget:
    """限制递归投影的深度、总项目数并检测当前身份环。"""

    active_ids: set[int] = field(default_factory=set)
    item_count: int = 0


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
        """让所有入口共享同一套结构验证、脱敏和递归冻结逻辑。"""
        inherited_reason = (
            self.unsupported_reason
            if isinstance(self.unsupported_reason, str) and self.unsupported_reason
            else "unsupported_projection" if self.supported is False else None
        )
        reason = inherited_reason
        budget = _JsonBudget()

        game_id = self.game_id if isinstance(self.game_id, str) else ""
        if not isinstance(self.game_id, str):
            reason = reason or "invalid_game_id"

        status_is_valid = (
            isinstance(self.status, str) and self.status in _GAME_STATUSES
        )
        status: GameStatus = self.status if status_is_valid else "running"
        if not status_is_valid:
            reason = reason or "invalid_status"

        winning_faction = self.winning_faction
        if winning_faction is not None and not isinstance(winning_faction, str):
            winning_faction = None
            reason = reason or "invalid_winning_faction"

        termination_reason = self.termination_reason
        if termination_reason is not None and not isinstance(termination_reason, str):
            termination_reason = None
            reason = reason or "invalid_termination_reason"

        steps = self.steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            steps = 0
            reason = reason or "invalid_steps"

        events: tuple[dict[str, Any], ...] = ()
        if not isinstance(self.events, (list, tuple)):
            reason = reason or "invalid_events_container"
        else:
            try:
                events = tuple(_normalize_event(event, budget) for event in self.events)
            except _ProjectionValueError as exc:
                reason = reason or str(exc)

        players: dict[str, dict[str, Any]] = {}
        if not isinstance(self.players, Mapping):
            reason = reason or "invalid_players_container"
        else:
            try:
                players = _normalize_players(self.players, budget)
            except _ProjectionValueError as exc:
                reason = reason or str(exc)

        deaths: tuple[dict[str, Any], ...] = ()
        if not isinstance(self.deaths, (list, tuple)):
            reason = reason or "invalid_deaths_container"
        else:
            try:
                deaths = tuple(_normalize_death(death, budget) for death in self.deaths)
            except _ProjectionValueError as exc:
                reason = reason or str(exc)

        metadata: dict[str, Any] = {}
        if not isinstance(self.metadata, Mapping):
            reason = reason or "invalid_metadata_container"
        else:
            try:
                metadata = _normalize_metadata(self.metadata, budget)
            except _ProjectionValueError as exc:
                reason = reason or str(exc)

        reason = reason or _unsupported_reason(
            game_id, players, status=status, winner=winning_faction,
        )
        object.__setattr__(self, "game_id", game_id)
        object.__setattr__(self, "events", tuple(_freeze_normalized(event) for event in events))
        object.__setattr__(self, "players", _freeze_normalized(players))
        object.__setattr__(self, "winning_faction", winning_faction)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "termination_reason", termination_reason)
        object.__setattr__(self, "deaths", tuple(_freeze_normalized(death) for death in deaths))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "supported", reason is None)
        object.__setattr__(self, "unsupported_reason", reason)
        object.__setattr__(self, "metadata", _freeze_normalized(metadata))

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
    games: Iterable[AcceptanceGameProjection | Mapping[str, Any] | Any],
) -> Sequence[Mapping[str, Any]]:
    """统一验收入口，并返回深度不可变的已验证游戏序列。"""
    return tuple(
        _freeze_normalized(project_acceptance_game(game).to_mapping())
        for game in games
    )


def projection_support(games: Sequence[Mapping[str, Any]]) -> tuple[bool, str | None]:
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
    structural_reason: str | None = None
    raw_players = getattr(source, "players", {})
    players: Any = raw_players
    if isinstance(raw_players, Mapping):
        players = {}
        try:
            for player_id, player in raw_players.items():
                players[player_id] = {
                    "id": getattr(player, "id", player_id),
                    "role": getattr(player, "role", ""),
                    "alive": getattr(player, "alive", False),
                    "faction": getattr(player, "faction", None),
                }
        except Exception:
            players = {}
            structural_reason = "invalid_player_entry"

    raw_events = getattr(source, "events", ())
    events: Any = raw_events
    if isinstance(raw_events, (list, tuple)):
        serialized_events: list[Any] = []
        try:
            for event in raw_events:
                payload = getattr(event, "payload", None)
                _normalize_json(payload, _JsonBudget())
                serialized_events.append(serialize_game_event(event))
            events = serialized_events
        except _ProjectionValueError as exc:
            events = ()
            structural_reason = (
                str(exc) if str(exc) in _JSON_BOUND_REASONS
                else "invalid_event_payload"
            )
        except (AttributeError, TypeError, ValueError):
            events = ()
            structural_reason = "invalid_event_entry"

    raw_deaths = getattr(source, "deaths", ())
    deaths: Any = raw_deaths
    if isinstance(raw_deaths, (list, tuple)):
        serialized_deaths: list[Any] = []
        try:
            for death in raw_deaths:
                serialized_batch, batch_failed = serialize_resolution_batch(
                    death.resolution_batch
                )
                serialized_deaths.append({
                    "player_id": death.player_id,
                    "reason": death.reason,
                    "timing": death.timing,
                    "resolution_batch": serialized_batch,
                    "resolution_batch_parse_failed": bool(
                        death.resolution_batch_parse_failed or batch_failed
                    ),
                    "source_player_id": death.source_player_id,
                    "can_leave_last_words": death.can_leave_last_words,
                    "triggered_skills": death.triggered_skills,
                })
            deaths = serialized_deaths
        except (AttributeError, TypeError, ValueError):
            deaths = ()
            structural_reason = structural_reason or "invalid_death_entry"

    winner = getattr(source, "winning_faction", None)
    raw_status = getattr(source, "status", "running")
    status = _derive_status(raw_status, winner, getattr(source, "phase", None))
    metadata = {
        "phase": getattr(source, "phase", None),
        "day_number": getattr(source, "day_number", 0),
        "night_number": getattr(source, "night_number", 0),
        **_hybrid_metadata(source, events if isinstance(events, (list, tuple)) else ()),
    }
    return AcceptanceGameProjection(
        game_id=getattr(source, "game_id", ""),
        events=events,
        players=players,
        winning_faction=winner,
        status=status,
        termination_reason=getattr(source, "termination_reason", None),
        deaths=deaths,
        steps=steps if steps is not None else getattr(source, "steps", 0) or 0,
        supported=structural_reason is None,
        unsupported_reason=structural_reason,
        metadata=metadata,
    )


def _from_mapping(source: Mapping[str, Any], *, steps: int | None) -> AcceptanceGameProjection:
    explicit_reason = source.get("_acceptance_projection_unsupported_reason")
    structural_reason: str | None = None
    if source.get("_acceptance_projection_supported") is False:
        structural_reason = (
            explicit_reason
            if isinstance(explicit_reason, str) and explicit_reason
            else "unsupported_projection"
        )
    if "events" not in source:
        structural_reason = structural_reason or "missing_events"
    winner = source.get("winning_faction")
    if winner is None:
        winner = source.get("winner")
    raw_status = source.get("status")
    status = (
        _derive_status(None, winner, source.get("phase"))
        if raw_status is None else raw_status
    )
    metadata = {key: source[key] for key in _METADATA_FIELDS if key in source}
    return AcceptanceGameProjection(
        game_id=source.get("game_id", ""),
        events=source.get("events", ()),
        players=source.get("players", {}),
        winning_faction=winner,
        status=status,
        termination_reason=source.get("termination_reason"),
        deaths=source.get("deaths", ()),
        steps=steps if steps is not None else source.get("steps") or 0,
        supported=structural_reason is None,
        unsupported_reason=structural_reason,
        metadata=metadata,
    )


def _normalize_event(value: Any, budget: _JsonBudget) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _ProjectionValueError("invalid_event_entry")
    event_type = value.get("type")
    payload = value.get("payload", {})
    if not isinstance(event_type, str) or not event_type or not isinstance(payload, Mapping):
        raise _ProjectionValueError("invalid_event_entry")
    _validate_repair_failure_history(event_type, payload)
    try:
        normalized = _normalize_json(value, budget)
    except _ProjectionValueError as exc:
        if str(exc) in _JSON_BOUND_REASONS:
            raise
        raise _ProjectionValueError("invalid_event_payload") from exc
    assert isinstance(normalized, dict)
    return normalized


def _validate_repair_failure_history(
    event_type: str,
    payload: Mapping[str, Any],
) -> None:
    """在冻结为 tuple 前严格校验可选的 JSON 修复历史列表。"""
    audits: list[Mapping[str, Any]] = []
    if event_type == "semantic_repair_audit":
        audits.append(payload)
    elif event_type == "action_trace_audit":
        trace = payload.get("action_trace")
        if isinstance(trace, Mapping):
            semantic = trace.get("semantic_repair_audit")
            if isinstance(semantic, Mapping):
                audits.append(semantic)

    for audit in audits:
        if "repair_failure_history" not in audit:
            continue
        # 已归一化快照中的 JSON list 会冻结为 tuple；外部原始输入必须是 list。
        expected_type = tuple if type(audit) is MappingProxyType else list
        if not _has_valid_repair_failure_history(
            audit,
            container_type=expected_type,
        ):
            raise _ProjectionValueError("invalid_semantic_repair_history")


def _normalize_players(
    value: Mapping[str, Any], budget: _JsonBudget,
) -> dict[str, dict[str, Any]]:
    """仅保留验收投影需要的玩家字段并验证其标量类型。"""
    projected: dict[str, dict[str, Any]] = {}
    for player_id, player in value.items():
        budget.item_count += 1
        if budget.item_count > _MAX_JSON_ITEMS:
            raise _ProjectionValueError("json_item_limit_exceeded")
        if not isinstance(player_id, str):
            raise _ProjectionValueError("invalid_player_id")
        if not isinstance(player, Mapping):
            raise _ProjectionValueError("invalid_player_entry")
        row = {key: player[key] for key in _PLAYER_FIELDS if key in player}
        row.setdefault("id", player_id)
        if not isinstance(row["id"], str):
            raise _ProjectionValueError("invalid_player_value")
        if "role" in row and not isinstance(row["role"], str):
            raise _ProjectionValueError("invalid_player_value")
        if "alive" in row and not isinstance(row["alive"], bool):
            raise _ProjectionValueError("invalid_player_value")
        if "faction" in row and row["faction"] is not None and not isinstance(row["faction"], str):
            raise _ProjectionValueError("invalid_player_value")
        projected[player_id] = row
    return projected


def _normalize_death(value: Any, budget: _JsonBudget) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _ProjectionValueError("invalid_death_entry")
    try:
        normalized = _normalize_json(value, budget)
    except _ProjectionValueError as exc:
        if str(exc) in _JSON_BOUND_REASONS:
            raise
        raise _ProjectionValueError("invalid_death_payload") from exc
    assert isinstance(normalized, dict)
    return normalized


def _normalize_metadata(
    value: Mapping[str, Any], budget: _JsonBudget,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _METADATA_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if key == "__source_path" and isinstance(item, Path):
            item = str(item)
        if key in {"phase", "__source_path", "hybrid_master_id", "hybrid_master_faction"}:
            if item is not None and not isinstance(item, str):
                raise _ProjectionValueError("invalid_metadata_value")
        elif key in {"day_number", "night_number"}:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise _ProjectionValueError("invalid_metadata_value")
        try:
            metadata[key] = _normalize_json(item, budget)
        except _ProjectionValueError as exc:
            if str(exc) in _JSON_BOUND_REASONS:
                raise
            raise _ProjectionValueError("invalid_metadata_value") from exc
    return metadata


def _normalize_json(
    value: Any,
    budget: _JsonBudget,
    *,
    depth: int = 0,
) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise _ProjectionValueError("json_depth_exceeded")
    budget.item_count += 1
    if budget.item_count > _MAX_JSON_ITEMS:
        raise _ProjectionValueError("json_item_limit_exceeded")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise _ProjectionValueError("non_finite_number")
    if isinstance(value, Enum):
        return _normalize_json(value.value, budget, depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in budget.active_ids:
            raise _ProjectionValueError("cyclic_json_value")
        budget.active_ids.add(identity)
        try:
            return {
                item.name: _normalize_json(
                    getattr(value, item.name), budget, depth=depth + 1,
                )
                for item in fields(value)
            }
        finally:
            budget.active_ids.remove(identity)
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in budget.active_ids:
            raise _ProjectionValueError("cyclic_json_value")
        budget.active_ids.add(identity)
        try:
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    raise _ProjectionValueError("non_string_mapping_key")
                return {
                    key: _normalize_json(item, budget, depth=depth + 1)
                    for key, item in value.items()
                }
            return [
                _normalize_json(item, budget, depth=depth + 1)
                for item in value
            ]
        finally:
            budget.active_ids.remove(identity)
    raise _ProjectionValueError("unsupported_json_value")


def _freeze_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze_normalized(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_normalized(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _derive_status(raw: Any, winner: Any, phase: Any) -> Any:
    if raw == "aborted":
        return "aborted"
    if (
        raw == "finished"
        or (isinstance(winner, str) and bool(winner))
        or phase == "finished"
    ):
        return "finished"
    if raw is not None and raw != "running":
        return raw
    return "running"


def _hybrid_metadata(source: Any, events: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    fields = {
        "hybrid_master_id": getattr(source, "hybrid_master_id", None),
        "hybrid_master_faction": getattr(source, "hybrid_master_faction", None),
        "hybrid_result": getattr(source, "hybrid_result", None),
    }
    for event in reversed(events):
        if isinstance(event, Mapping) and event.get("type") == "victory":
            payload = event.get("payload") or {}
            if isinstance(payload, Mapping):
                for key in fields:
                    if fields[key] is None:
                        fields[key] = payload.get(key)
            break
    return fields


def _unsupported_reason(
    game_id: str,
    players: Mapping[str, Mapping[str, Any]],
    *,
    status: GameStatus,
    winner: Any,
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


__all__ = [
    "AcceptanceGameProjection", "normalize_acceptance_games",
    "normalize_quality_score", "project_acceptance_game",
    "projection_support",
]
