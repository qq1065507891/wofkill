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

from dataclasses import dataclass
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
_TRANSACTION_SEAL = object()
_STAGE_PATHS = {
    stage: tuple(list(ReflectionStage)[:index + 1])
    for index, stage in enumerate(ReflectionStage)
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


@dataclass(frozen=True, slots=True, init=False)
class PlayerReflectionTransaction:
    """一名玩家的反思事务快照；公开构造器只能创建初态。"""

    player_id: str
    decision_id: str
    stage: ReflectionStage
    failure_stage: str | None
    failure_code: str | None
    verified_claim_ids: tuple[str, ...]
    verified_lesson_ids: tuple[str, ...]
    entry_id: str | None
    _stage_path: tuple[ReflectionStage, ...]
    _seal: object

    def __init__(self, player_id: str, decision_id: str) -> None:
        """创建 not_requested 初态，禁止调用方直接注入后续状态。"""
        values = {
            "player_id": player_id,
            "decision_id": decision_id,
            "stage": ReflectionStage.NOT_REQUESTED,
            "failure_stage": None,
            "failure_code": None,
            "verified_claim_ids": (),
            "verified_lesson_ids": (),
            "entry_id": None,
            "_stage_path": (ReflectionStage.NOT_REQUESTED,),
            "_seal": _TRANSACTION_SEAL,
        }
        for field, value in values.items():
            object.__setattr__(self, field, value)
        self.validate()

    def validate(self) -> None:
        """独立验证阶段、字段组合、失败位置和完整转换路径。"""
        if getattr(self, "_seal", None) is not _TRANSACTION_SEAL:
            raise ValueError("reflection transaction provenance is missing")
        if not self.player_id or not self.decision_id:
            raise ValueError("player_id and decision_id are required")
        if not isinstance(self.stage, ReflectionStage):
            raise ValueError("invalid reflection stage")
        if self._stage_path != _STAGE_PATHS[self.stage]:
            raise ValueError("reflection transaction stage provenance is invalid")
        if (self.failure_stage is None) != (self.failure_code is None):
            raise ValueError("failure_stage and failure_code must be provided together")
        if self.failure_stage is not None and (
            not self.failure_stage or not self.failure_code
        ):
            raise ValueError("failure_stage and failure_code must be non-empty")
        if self.failure_stage is not None:
            expected_failure = _NEXT_STAGE.get(self.stage)
            if (
                expected_failure is None
                or self.failure_stage != expected_failure.value
            ):
                raise ValueError("failure_stage must name the next uncompleted stage")

        if self.stage in {
            ReflectionStage.NOT_REQUESTED,
            ReflectionStage.GENERATED,
            ReflectionStage.SCHEMA_VALIDATED,
        } and (self.verified_claim_ids or self.verified_lesson_ids or self.entry_id):
            raise ValueError("early reflection stage cannot carry verified identities")
        if self.stage is ReflectionStage.FACTS_VERIFIED and (
            self.verified_lesson_ids or self.entry_id
        ):
            raise ValueError("facts_verified cannot carry lesson or entry identity")
        if self.stage in {
            ReflectionStage.LESSONS_VERIFIED,
            ReflectionStage.PERSISTED,
        } and (not self.verified_claim_ids or not self.verified_lesson_ids):
            raise ValueError(
                "verified lesson requires a non-empty claim and lesson identity chain"
            )
        if self.stage is ReflectionStage.PERSISTED:
            if not self.entry_id:
                raise ValueError("persisted reflection requires entry_id")
        elif self.entry_id is not None:
            raise ValueError("entry_id belongs only to persisted reflection")
        _identifiers(self.verified_claim_ids)
        _identifiers(self.verified_lesson_ids)

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
        changes: dict[str, object] = {
            "stage": next_stage,
            "stage_path": (*self._stage_path, next_stage),
        }
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
        candidate = object.__new__(type(self))
        values = {
            "player_id": self.player_id,
            "decision_id": self.decision_id,
            "stage": changes.get("stage", self.stage),
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "verified_claim_ids": changes.get(
                "verified_claim_ids", self.verified_claim_ids,
            ),
            "verified_lesson_ids": changes.get(
                "verified_lesson_ids", self.verified_lesson_ids,
            ),
            "entry_id": changes.get("entry_id", self.entry_id),
            "_stage_path": changes.get("stage_path", self._stage_path),
            "_seal": _TRANSACTION_SEAL,
        }
        for field, value in values.items():
            object.__setattr__(candidate, field, value)
        candidate.validate()
        return candidate

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
        candidate = object.__new__(type(self))
        values = {
            "player_id": self.player_id,
            "decision_id": self.decision_id,
            "stage": self.stage,
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "verified_claim_ids": self.verified_claim_ids,
            "verified_lesson_ids": self.verified_lesson_ids,
            "entry_id": self.entry_id,
            "_stage_path": self._stage_path,
            "_seal": _TRANSACTION_SEAL,
        }
        for field, value in values.items():
            object.__setattr__(candidate, field, value)
        candidate.validate()
        return candidate

    def to_payload(self) -> dict[str, object]:
        """生成可安全写入 moderator-only 事件的事务字段。"""
        self.validate()
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
    raw_items = tuple(entries)
    if not transaction_run:
        return ReflectionTransactionResult("not_run", False, 0, 0, 0)

    items: list[PlayerReflectionTransaction] = []
    invalid_count = 0
    seen_objects: set[int] = set()
    seen_players: set[str] = set()
    seen_decisions: set[str] = set()
    seen_entries: set[str] = set()
    for item in raw_items:
        try:
            if not isinstance(item, PlayerReflectionTransaction):
                raise ValueError("unexpected reflection transaction type")
            item.validate()
            object_id = id(item)
            if (
                object_id in seen_objects
                or item.player_id in seen_players
                or item.decision_id in seen_decisions
                or (
                    item.entry_id is not None
                    and item.entry_id in seen_entries
                )
            ):
                raise ValueError("duplicate reflection transaction identity")
        except (AttributeError, TypeError, ValueError):
            invalid_count += 1
            continue
        seen_objects.add(object_id)
        seen_players.add(item.player_id)
        seen_decisions.add(item.decision_id)
        if item.entry_id is not None:
            seen_entries.add(item.entry_id)
        items.append(item)

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
            "no_valid_entries", False, 0, 0, len(failures) + invalid_count,
        )

    all_valid_persisted = len(persisted) == len(valid)
    all_other_entries_attributed = all(
        item.stage is ReflectionStage.PERSISTED
        or (item.failure_stage is not None and item.failure_code is not None)
        for item in items
    )
    if invalid_count or (persistence_attempted and not all_valid_persisted):
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
        len(failures) + invalid_count,
    )


__all__ = [
    "PlayerReflectionTransaction",
    "ReflectionStage",
    "ReflectionTransactionResult",
    "ReflectionTransactionStatus",
    "ReflectionTransitionError",
    "summarize_reflection_transaction",
]
