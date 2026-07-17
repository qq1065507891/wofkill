# -*- coding: utf-8 -*-
"""
定义赛后反思的逐玩家事务状态机与局级结果汇总。

作者: Project contributors
创建日期: 2026-07-17
修改日期: 2026-07-18

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


def _validate_transaction_values(
    *,
    player_id: str,
    decision_id: str,
    stage: ReflectionStage,
    failure_stage: str | None,
    failure_code: str | None,
    verified_claim_ids: tuple[str, ...],
    verified_lesson_ids: tuple[str, ...],
    entry_id: str | None,
    stage_path: tuple[ReflectionStage, ...],
) -> None:
    """验证快照的数据约束，不把可复制的模块 seal 当作来源证明。"""
    if not player_id or not decision_id:
        raise ValueError("player_id and decision_id are required")
    if not isinstance(stage, ReflectionStage):
        raise ValueError("invalid reflection stage")
    if stage_path != _STAGE_PATHS[stage]:
        raise ValueError("reflection transaction stage provenance is invalid")
    if (failure_stage is None) != (failure_code is None):
        raise ValueError("failure_stage and failure_code must be provided together")
    if failure_stage is not None and (not failure_stage or not failure_code):
        raise ValueError("failure_stage and failure_code must be non-empty")
    if failure_stage is not None:
        expected_failure = _NEXT_STAGE.get(stage)
        if expected_failure is None or failure_stage != expected_failure.value:
            raise ValueError("failure_stage must name the next uncompleted stage")
    if stage in {
        ReflectionStage.NOT_REQUESTED,
        ReflectionStage.GENERATED,
        ReflectionStage.SCHEMA_VALIDATED,
    } and (verified_claim_ids or verified_lesson_ids or entry_id):
        raise ValueError("early reflection stage cannot carry verified identities")
    if stage is ReflectionStage.FACTS_VERIFIED and (verified_lesson_ids or entry_id):
        raise ValueError("facts_verified cannot carry lesson or entry identity")
    if stage in {
        ReflectionStage.LESSONS_VERIFIED,
        ReflectionStage.PERSISTED,
    } and (not verified_claim_ids or not verified_lesson_ids):
        raise ValueError(
            "verified lesson requires a non-empty claim and lesson identity chain"
        )
    if stage is ReflectionStage.PERSISTED:
        if not entry_id:
            raise ValueError("persisted reflection requires entry_id")
    elif entry_id is not None:
        raise ValueError("entry_id belongs only to persisted reflection")
    _identifiers(verified_claim_ids)
    _identifiers(verified_lesson_ids)


def _build_transaction_api():
    """创建闭包内不可变快照类型以及公开的初态工厂。"""
    class _TransactionSnapshot(tuple):
        __slots__ = ()

        player_id = property(lambda self: self[0])
        decision_id = property(lambda self: self[1])
        stage = property(lambda self: self[2])
        failure_stage = property(lambda self: self[3])
        failure_code = property(lambda self: self[4])
        verified_claim_ids = property(lambda self: self[5])
        verified_lesson_ids = property(lambda self: self[6])
        entry_id = property(lambda self: self[7])
        _stage_path = property(lambda self: self[8])

        def validate(self) -> None:
            _validate_transaction_values(
                player_id=self.player_id,
                decision_id=self.decision_id,
                stage=self.stage,
                failure_stage=self.failure_stage,
                failure_code=self.failure_code,
                verified_claim_ids=self.verified_claim_ids,
                verified_lesson_ids=self.verified_lesson_ids,
                entry_id=self.entry_id,
                stage_path=self._stage_path,
            )

        def advance(
            self,
            next_stage: ReflectionStage,
            *,
            verified_claim_ids: Iterable[str] | None = None,
            verified_lesson_ids: Iterable[str] | None = None,
            entry_id: str | None = None,
        ):
            """只允许相邻推进，并在对应边界绑定验证身份。"""
            self.validate()
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
            claim_ids = self.verified_claim_ids
            lesson_ids = self.verified_lesson_ids
            persisted_entry_id = self.entry_id
            if verified_claim_ids is not None:
                if next_stage is not ReflectionStage.FACTS_VERIFIED:
                    raise ValueError("verified_claim_ids belong to facts_verified")
                claim_ids = _identifiers(verified_claim_ids)
            if verified_lesson_ids is not None:
                if next_stage is not ReflectionStage.LESSONS_VERIFIED:
                    raise ValueError("verified_lesson_ids belong to lessons_verified")
                lesson_ids = _identifiers(verified_lesson_ids)
                if not lesson_ids:
                    raise ValueError("lessons_verified requires at least one lesson")
            if next_stage is ReflectionStage.LESSONS_VERIFIED and not lesson_ids:
                raise ValueError("lessons_verified requires at least one lesson")
            if next_stage is ReflectionStage.PERSISTED:
                if not isinstance(entry_id, str) or not entry_id:
                    raise ValueError("persisted reflection requires entry_id")
                persisted_entry_id = entry_id
            elif entry_id is not None:
                raise ValueError("entry_id belongs only to persisted")
            return _make_snapshot(
                self.player_id,
                self.decision_id,
                next_stage,
                None,
                None,
                claim_ids,
                lesson_ids,
                persisted_entry_id,
                (*self._stage_path, next_stage),
            )

        def fail(self, *, failure_stage: str, failure_code: str):
            """在最后成功阶段保留明确失败边界。"""
            self.validate()
            if self.stage is ReflectionStage.PERSISTED:
                raise ReflectionTransitionError("persisted reflection cannot fail")
            if not failure_stage or not failure_code:
                raise ValueError("failure_stage and failure_code must be non-empty")
            return _make_snapshot(
                self.player_id,
                self.decision_id,
                self.stage,
                failure_stage,
                failure_code,
                self.verified_claim_ids,
                self.verified_lesson_ids,
                self.entry_id,
                self._stage_path,
            )

        def to_payload(self) -> dict[str, object]:
            """生成可安全写入 moderator-only 事件的交易字段。"""
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

        def __reduce__(self):
            return (_restore_reflection_transaction, (self.to_payload(),))

    def _make_snapshot(
        player_id: str,
        decision_id: str,
        stage: ReflectionStage,
        failure_stage: str | None,
        failure_code: str | None,
        verified_claim_ids: tuple[str, ...],
        verified_lesson_ids: tuple[str, ...],
        entry_id: str | None,
        stage_path: tuple[ReflectionStage, ...],
    ):
        _validate_transaction_values(
            player_id=player_id,
            decision_id=decision_id,
            stage=stage,
            failure_stage=failure_stage,
            failure_code=failure_code,
            verified_claim_ids=verified_claim_ids,
            verified_lesson_ids=verified_lesson_ids,
            entry_id=entry_id,
            stage_path=stage_path,
        )
        return tuple.__new__(_TransactionSnapshot, (
            player_id,
            decision_id,
            stage,
            failure_stage,
            failure_code,
            verified_claim_ids,
            verified_lesson_ids,
            entry_id,
            stage_path,
        ))

    class _TransactionFacadeMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            return type(instance) is _TransactionSnapshot

    class PlayerReflectionTransaction(metaclass=_TransactionFacadeMeta):
        """创建玩家反思初态；后续状态由不可变快照逐级产生。"""

        def __new__(cls, player_id: str, decision_id: str):
            return _make_snapshot(
                player_id,
                decision_id,
                ReflectionStage.NOT_REQUESTED,
                None,
                None,
                (),
                (),
                None,
                (ReflectionStage.NOT_REQUESTED,),
            )

    def _is_snapshot(value: object) -> bool:
        return type(value) is _TransactionSnapshot

    return PlayerReflectionTransaction, _is_snapshot


PlayerReflectionTransaction, _is_transaction_snapshot = _build_transaction_api()


def _restore_reflection_transaction(payload: dict[str, object]):
    """由可序列化字段重放合法转换，不直接恢复任意对象状态。"""
    if not isinstance(payload, dict):
        raise ValueError("reflection transaction payload must be a mapping")
    player_id = payload.get("player_id")
    decision_id = payload.get("decision_id")
    if not isinstance(player_id, str) or not isinstance(decision_id, str):
        raise ValueError("player_id and decision_id are required")
    try:
        target_stage = ReflectionStage(payload.get("transaction_state"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid reflection stage") from exc
    transaction = PlayerReflectionTransaction(player_id, decision_id)
    if target_stage is not ReflectionStage.NOT_REQUESTED:
        transaction = transaction.advance(ReflectionStage.GENERATED)
    if target_stage in {
        ReflectionStage.SCHEMA_VALIDATED,
        ReflectionStage.FACTS_VERIFIED,
        ReflectionStage.LESSONS_VERIFIED,
        ReflectionStage.PERSISTED,
    }:
        transaction = transaction.advance(ReflectionStage.SCHEMA_VALIDATED)
    if target_stage in {
        ReflectionStage.FACTS_VERIFIED,
        ReflectionStage.LESSONS_VERIFIED,
        ReflectionStage.PERSISTED,
    }:
        transaction = transaction.advance(
            ReflectionStage.FACTS_VERIFIED,
            verified_claim_ids=payload.get("verified_claim_ids") or (),
        )
    if target_stage in {ReflectionStage.LESSONS_VERIFIED, ReflectionStage.PERSISTED}:
        transaction = transaction.advance(
            ReflectionStage.LESSONS_VERIFIED,
            verified_lesson_ids=payload.get("verified_lesson_ids") or (),
        )
    if target_stage is ReflectionStage.PERSISTED:
        transaction = transaction.advance(
            ReflectionStage.PERSISTED,
            entry_id=payload.get("entry_id"),
        )
    failure_stage = payload.get("failure_stage")
    failure_code = payload.get("failure_code")
    if failure_stage is not None or failure_code is not None:
        if not isinstance(failure_stage, str) or not isinstance(failure_code, str):
            raise ValueError("failure_stage and failure_code must be strings")
        transaction = transaction.fail(
            failure_stage=failure_stage,
            failure_code=failure_code,
        )
    return transaction


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
            if not _is_transaction_snapshot(item):
                raise ValueError("unexpected reflection transaction type")
            item.validate()
            object_id = id(item)
            if (
                object_id in seen_objects
                or item.player_id in seen_players
                or item.decision_id in seen_decisions
                or (item.entry_id is not None and item.entry_id in seen_entries)
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
