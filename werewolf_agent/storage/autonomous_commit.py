# -*- coding: utf-8 -*-
"""
提供自主玩家 CommitTurn 的仓储能力协议、哈希和事件绑定辅助函数。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Protocol, cast

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameEvent
from werewolf_agent.player_agents.contracts.errors import ValidationErrorCode
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
from werewolf_agent.player_agents.contracts.transactions import (
    CommitResult,
    CommitTurnRequest,
    EventCandidate,
    ProjectionOutboxRecord,
)


class AutonomousCommitUnsupported(RuntimeError):
    """仓储未声明自主玩家提交能力。"""

    code = ValidationErrorCode.UNKNOWN_CAPABILITY


class StaleCommitError(RuntimeError):
    """提交所依据的游戏 revision 已过期。"""

    code = ValidationErrorCode.STALE_READ_SET


class IdempotencyConflictError(RuntimeError):
    """幂等键已绑定到不同的提交请求。"""

    code = ValidationErrorCode.IDEMPOTENCY_CONFLICT


class CommitTransactionError(RuntimeError):
    """CommitTurn 事务失败且已回滚。"""


class AutonomousCommitRepository(Protocol):
    """自主玩家运行时启用前必须满足的显式能力接口。"""

    def supports_autonomous_commit(self) -> bool: ...

    def commit_turn(self, request: CommitTurnRequest) -> CommitResult: ...

    def load_game_revision(self, game_id: str) -> int: ...

    def load_outbox(self, game_id: str) -> list[ProjectionOutboxRecord]: ...


def request_hash(request: CommitTurnRequest) -> str:
    """使用稳定 JSON 表示计算提交请求的内容哈希。"""
    canonical = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_committed_event(
    game_id: str,
    candidate: EventCandidate,
    revision: int,
) -> GameEvent:
    """为候选事件分配唯一身份和 authoritative revision。"""
    visibility = (
        EventVisibility.from_legacy(candidate.visibility)
        if candidate.visibility is not None
        else event_visibility(
            GameEvent(type=candidate.type, payload=dict(candidate.payload)),
        )
    )
    return GameEvent(
        type=candidate.type,
        payload=dict(candidate.payload),
        visibility=visibility,
        event_id=f"{game_id}:e{revision:06d}",
        sequence_number=revision,
        occurred_at=datetime.now(timezone.utc),
        game_id=game_id,
        schema_version="2",
    )


def bind_public_record(
    record: PublicSpeechRecord | None,
    revision: int,
) -> PublicSpeechRecord | None:
    """将仓储分配的 revision 写入公共语义记录。"""
    if record is None:
        return None
    return record.model_copy(update={"committed_revision": revision})


def build_commit_result(
    request: CommitTurnRequest,
    digest: str,
    revision: int,
    event: GameEvent,
    record: PublicSpeechRecord | None,
) -> CommitResult:
    """从已绑定的提交材料构造稳定结果。"""
    return CommitResult(
        game_id=request.game_id,
        turn_id=request.turn_id,
        idempotency_key=request.idempotency_key,
        committed_revision=revision,
        event_id=event.event_id or "",
        public_record_id=record.record_id if record is not None else None,
        audit_ids=tuple(item.audit_id for item in request.critical_audit_records),
        outbox_ids=tuple(item.outbox_id for item in request.projection_outbox_records),
        request_hash=digest,
    )


def require_autonomous_commit_repository(
    repository: object,
) -> AutonomousCommitRepository:
    """只接受显式声明 capability 的仓储对象。"""
    supports = getattr(repository, "supports_autonomous_commit", None)
    if not callable(supports) or not supports():
        raise AutonomousCommitUnsupported(
            "repository does not support autonomous CommitTurn transactions",
        )
    return cast(AutonomousCommitRepository, repository)


__all__ = [
    "AutonomousCommitRepository",
    "AutonomousCommitUnsupported",
    "CommitTransactionError",
    "IdempotencyConflictError",
    "StaleCommitError",
    "bind_public_record",
    "build_commit_result",
    "build_committed_event",
    "request_hash",
    "require_autonomous_commit_repository",
]
