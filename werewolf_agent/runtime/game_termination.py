# -*- coding: utf-8 -*-
"""
集中管理游戏完成、中止与最小化应急持久化。

作者: Project contributors
创建日期: 2026-07-16

使用示例:
    >>> finish_game(GameState(game_id="g1", winning_faction="good"))
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
import uuid

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.event_metadata import new_game_event


_SAFE_GAME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ABORT_EVENT_TYPE = "game_aborted"


def finish_game(state: GameState) -> GameState:
    """把有明确胜者的游戏转为不可变完成终态。"""
    if state.status == "finished":
        return state
    if state.status == "aborted":
        raise RuntimeError("terminal game state cannot transition from aborted to finished")
    if not state.winning_faction:
        raise ValueError("finished game requires a winner")
    return replace(state, status="finished", termination_reason=None)


def abort_game(
    state: GameState,
    *,
    reason: str,
    last_node: str | None,
    step: int,
    exception: BaseException | None = None,
    now: datetime | None = None,
) -> GameState:
    """记录一次 moderator-only V2 事件并转为中止终态。"""
    if state.status == "aborted":
        validate_aborted_game(state)
        return state
    if state.status == "finished":
        raise RuntimeError("terminal game state cannot transition from finished to aborted")
    if not reason:
        raise ValueError("aborted game requires termination_reason")
    occurred_at = now or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "termination_reason": reason,
        "last_node": last_node,
        "phase": state.phase,
        "step": int(step),
        "exception_type": type(exception).__name__ if exception is not None else None,
    }
    event = new_game_event(
        state,
        _ABORT_EVENT_TYPE,
        payload,
        visibility=EventVisibility.MODERATOR_ONLY,
        now=occurred_at,
    )
    return replace(
        state,
        status="aborted",
        termination_reason=reason,
        winning_faction=None,
        events=[*state.events, event],
    )


def validate_aborted_game(state: GameState) -> None:
    """对 runtime 新写 aborted 终态执行完整、失败关闭的校验。"""
    if state.status != "aborted" or not state.termination_reason:
        raise ValueError("runtime aborted game requires termination_reason")
    events = [item for item in state.events if item.type == _ABORT_EVENT_TYPE]
    if len(events) != 1:
        raise ValueError("aborted game requires exactly one game_aborted event")
    event = events[0]
    if event.visibility is not EventVisibility.MODERATOR_ONLY:
        raise ValueError("game_aborted event must be moderator_only")
    if (
        event.schema_version != "2"
        or event.event_id is None
        or event.sequence_number is None
        or event.occurred_at is None
        or event.game_id is None
    ):
        raise ValueError("game_aborted event must have complete V2 metadata")
    if event.game_id != state.game_id:
        raise ValueError("game_aborted event game_id must match state")
    sequence = event.sequence_number
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
        or event.event_id != f"{state.game_id}:e{sequence:06d}"
    ):
        raise ValueError("game_aborted event has invalid V2 identity")
    required = {
        "termination_reason", "last_node", "phase", "step", "exception_type",
    }
    missing = required.difference(event.payload)
    if missing:
        raise ValueError(
            "game_aborted event missing payload fields: " + ", ".join(sorted(missing))
        )
    if event.payload["termination_reason"] != state.termination_reason:
        raise ValueError("game_aborted termination_reason must match state")
    if event.payload["phase"] != state.phase:
        raise ValueError("game_aborted phase must match state")
    step = event.payload["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("game_aborted step must be a non-negative integer")
    for field_name in ("last_node", "exception_type"):
        value = event.payload[field_name]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"game_aborted {field_name} must be a string or null")


def emergency_abort_payload(state: GameState) -> dict[str, Any]:
    """从中止终态生成不包含 prompt、角色或私密事件的白名单。"""
    validate_aborted_game(state)
    event = next(
        (item for item in reversed(state.events) if item.type == _ABORT_EVENT_TYPE),
        None,
    )
    if event is None:
        raise ValueError("aborted game is missing game_aborted event")
    return {
        "game_id": state.game_id,
        "status": state.status,
        "termination_reason": state.termination_reason,
        "last_node": event.payload.get("last_node"),
        "phase": event.payload.get("phase"),
        "day_number": state.day_number,
        "night_number": state.night_number,
        "step": event.payload.get("step"),
        "exception_type": event.payload.get("exception_type"),
        "occurred_at": (
            event.occurred_at.isoformat() if event.occurred_at is not None else None
        ),
    }


def write_emergency_abort(state: GameState, directory: str | Path) -> Path:
    """在受信根目录内原子写入最小化中止产物。"""
    if (
        not isinstance(state.game_id, str)
        or ".." in state.game_id
        or _SAFE_GAME_ID.fullmatch(state.game_id) is None
    ):
        raise ValueError("invalid game_id for emergency artifact")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = _resolve_within(root, root / f"emergency_abort_{state.game_id}.json")
    temporary = _resolve_within(root, root / f"{target.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(emergency_abort_payload(state), ensure_ascii=False, indent=2)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _resolve_within(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError("emergency artifact path is outside configured directory")
    return resolved


__all__ = [
    "abort_game",
    "emergency_abort_payload",
    "finish_game",
    "validate_aborted_game",
    "write_emergency_abort",
]
