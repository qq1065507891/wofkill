# -*- coding: utf-8 -*-
"""
定义赛后反思的逐玩家事务状态机与局级结果汇总。

作者: Project contributors
创建日期: 2026-07-17

使用示例:
    >>> entry = PlayerReflectionTransaction("p01", "reflection:g1:p01")
    >>> entry.advance(ReflectionStage.GENERATED).stage.value
    'generated'
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Literal


class ReflectionStage(str, Enum):
    """逐玩家反思只能按此顺序单向推进。"""

    NOT_REQUESTED = "not_requested"
    GENERATED = "generated"
    SCHEMA_VALIDATED = "schema_validated"
    FACTS_VERIFIED = "facts_verified"
    LESSONS_VERIFIED = "lessons_verified"
    PERSISTED = "persisted"


ReflectionTransactionStatus = Literal[
    "complete",
    "partial",
    "no_valid_entries",
    "persistence_failed",
    "not_run",
]


class ReflectionTransitionError(ValueError):
    """反思状态发生跳级、倒退或完成后重复推进。"""


_NEXT_STAGE = {
    ReflectionStage.NOT_REQUESTED: ReflectionStage.GENERATED,
    ReflectionStage.GENERATED: ReflectionStage.SCHEMA_VALIDATED,
    ReflectionStage.SCHEMA_VALIDATED: ReflectionStage.FACTS_VERIFIED,
    ReflectionStage.FACTS_VERIFIED: ReflectionStage.LESSONS_VERIFIED,
    ReflectionStage.LESSONS_VERIFIED: ReflectionStage.PERSISTED,
}


def _identifiers(values: Iterable[str]) -> tuple[str, ...]:
    """保序去重并拒绝空标识，确保身份链可精确比较。"""
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("reflection identifiers must be non-empty strings")
        if value not in result:
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class PlayerReflectionTransaction:
    """一名玩家的反思事务快照。"""

    player_id: str
    decision_id: str
    stage: ReflectionStage = ReflectionStage.NOT_REQUESTED
    failure_stage: str | None = None
    failure_code: str | None = None
    verified_claim_ids: tuple[str, ...] = ()
    verified_lesson_ids: tuple[str, ...] = ()
    entry_id: str | None = None

    def __post_init__(self) -> None:
        if not self.player_id or not self.decision_id:
            raise ValueError("player_id and decision_id are required")
        if (self.failure_stage is None) != (self.failure_code is None):
            raise ValueError("failure_stage and failure_code must be provided together")
        if self.failure_stage is not None and (
            not self.failure_stage or not self.failure_code
        ):
            raise ValueError("failure_stage and failure_code must be non-empty")
        if self.stage is ReflectionStage.PERSISTED and not self.entry_id:
            raise ValueError("persisted reflection requires entry_id")

    def advance(
        self,
        next_stage: ReflectionStage,
        *,
        verified_claim_ids: Iterable[str] | None = None,
        verified_lesson_ids: Iterable[str] | None = None,
        entry_id: str | None = None,
    ) -> PlayerReflectionTransaction:
        """只允许推进到紧邻下一状态，并在对应边界绑定身份。"""
        expected = _NEXT_STAGE.get(self.stage)
        if next_stage is not expected:
            raise ReflectionTransitionError(
                "illegal reflection transition: "
                f"{self.stage.value} -> {next_stage.value}"
            )
        if self.failure_stage is not None:
            raise ReflectionTransitionError(
                "illegal reflection transition: failed entry cannot advance"
            )
        changes: dict[str, object] = {"stage": next_stage}
        if verified_claim_ids is not None:
            if next_stage is not ReflectionStage.FACTS_VERIFIED:
                raise ValueError("verified_claim_ids belong to facts_verified")
            changes["verified_claim_ids"] = _identifiers(verified_claim_ids)
        if verified_lesson_ids is not None:
            if next_stage is not ReflectionStage.LESSONS_VERIFIED:
                raise ValueError("verified_lesson_ids belong to lessons_verified")
            lesson_ids = _identifiers(verified_lesson_ids)
            if not lesson_ids:
                raise ValueError("lessons_verified requires at least one lesson")
            changes["verified_lesson_ids"] = lesson_ids
        if next_stage is ReflectionStage.LESSONS_VERIFIED and (
            verified_lesson_ids is None and not self.verified_lesson_ids
        ):
            raise ValueError("lessons_verified requires at least one lesson")
        if next_stage is ReflectionStage.PERSISTED:
            if not isinstance(entry_id, str) or not entry_id:
                raise ValueError("persisted reflection requires entry_id")
            changes["entry_id"] = entry_id
        elif entry_id is not None:
            raise ValueError("entry_id belongs to persisted")
        return replace(self, **changes)

    def fail(
        self,
        *,
        failure_stage: str,
        failure_code: str,
    ) -> PlayerReflectionTransaction:
        """在当前最后成功状态保留明确失败边界。"""
        if self.stage is ReflectionStage.PERSISTED:
            raise ReflectionTransitionError("persisted reflection cannot fail")
        if not failure_stage or not failure_code:
            raise ValueError("failure_stage and failure_code must be non-empty")
        return replace(
            self,
            failure_stage=failure_stage,
            failure_code=failure_code,
        )

    def to_payload(self) -> dict[str, object]:
        """生成可安全写入 moderator-only 事件的事务字段。"""
        return {
            "player_id": self.player_id,
            "decision_id": self.decision_id,
            "transaction_state": self.stage.value,
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "verified_claim_ids": list(self.verified_claim_ids),
            "verified_lesson_ids": list(self.verified_lesson_ids),
            "entry_id": self.entry_id,
        }


@dataclass(frozen=True)
class ReflectionTransactionResult:
    """局级反思事务结果。"""

    status: ReflectionTransactionStatus
    persistence_complete: bool
    valid_entry_count: int
    persisted_entry_count: int
    failure_count: int


def summarize_reflection_transaction(
    entries: Iterable[PlayerReflectionTransaction],
    *,
    transaction_run: bool = True,
    persistence_attempted: bool = False,
) -> ReflectionTransactionResult:
    """汇总非真空事务；空集合永远不能成为成功证明。"""
    items = tuple(entries)
    if not transaction_run:
        return ReflectionTransactionResult("not_run", False, 0, 0, 0)

    valid = tuple(
        item for item in items
        if item.stage in {ReflectionStage.LESSONS_VERIFIED, ReflectionStage.PERSISTED}
    )
    persisted = tuple(
        item for item in valid if item.stage is ReflectionStage.PERSISTED
    )
    failures = tuple(item for item in items if item.failure_stage is not None)
    if not valid:
        return ReflectionTransactionResult(
            "no_valid_entries", False, 0, 0, len(failures),
        )

    all_valid_persisted = len(persisted) == len(valid)
    all_other_entries_attributed = all(
        item.stage is ReflectionStage.PERSISTED
        or (item.failure_stage is not None and item.failure_code is not None)
        for item in items
    )
    if persistence_attempted and not all_valid_persisted:
        status: ReflectionTransactionStatus = "persistence_failed"
        complete = False
    elif persistence_attempted and all_valid_persisted and all_other_entries_attributed:
        status = "complete" if len(persisted) == len(items) else "partial"
        complete = bool(persisted)
    else:
        status = "partial"
        complete = False
    return ReflectionTransactionResult(
        status,
        complete,
        len(valid),
        len(persisted),
        len(failures),
    )


__all__ = [
    "PlayerReflectionTransaction",
    "ReflectionStage",
    "ReflectionTransactionResult",
    "ReflectionTransactionStatus",
    "ReflectionTransitionError",
    "summarize_reflection_transaction",
]
