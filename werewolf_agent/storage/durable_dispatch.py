# -*- coding: utf-8 -*-
"""
定义自主玩家 durable dispatch 的仓储能力、恢复解析器与协调器。

该模块只描述显式 capability 和恢复协议，不负责任何具体内存或数据库持久化。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol, cast

from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchRecoveryPolicy,
    DispatchResultDisposition,
    DispatchResultRecord,
)


class DurableDispatchError(RuntimeError):
    """所有 durable dispatch 存储错误的稳定基类。"""

    code: ClassVar[str] = "durable_dispatch_error"


class DurableDispatchUnsupported(DurableDispatchError):
    """仓储未显式声明 durable dispatch capability。"""

    code = "durable_dispatch_unsupported"


class DispatchNotFound(DurableDispatchError):
    """请求的 dispatch ID 不存在。"""

    code = "dispatch_not_found"


class DispatchStateConflict(DurableDispatchError):
    """dispatch 的 CAS state version 已过期。"""

    code = "dispatch_state_conflict"


class DispatchInvalidTransition(DurableDispatchError):
    """dispatch 状态机不允许当前迁移。"""

    code = "dispatch_invalid_transition"


class DispatchIdempotencyConflict(DurableDispatchError):
    """dispatch ID 或 provider 幂等键发生冲突。"""

    code = "dispatch_idempotency_conflict"


class DispatchLeaseMismatch(DurableDispatchError):
    """结果绑定到不同 lease。"""

    code = "dispatch_lease_mismatch"


class DispatchResultConflict(DurableDispatchError):
    """重复结果与已持久化结果不一致。"""

    code = "dispatch_result_conflict"


class DispatchRecoveryBlocked(DurableDispatchError):
    """游戏仍有未决外部 dispatch，不能开始新的 dispatch。"""

    code = "dispatch_recovery_blocked"


class DispatchTransactionError(DurableDispatchError):
    """仓储事务失败。"""

    code = "dispatch_transaction_error"


class DurableDispatchRepository(Protocol):
    """实现 durable dispatch 的仓储必须显式满足的接口。"""

    def supports_durable_dispatch(self) -> bool: ...

    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt: ...

    def mark_dispatching(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt: ...

    def mark_dispatched(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt: ...

    def record_result(
        self,
        dispatch_id: str,
        expected_version: int,
        result: DispatchResultRecord,
    ) -> DispatchResultDisposition: ...

    def cancel_dispatch(
        self,
        dispatch_id: str,
        expected_version: int,
        reason_code: str,
    ) -> DispatchAttempt: ...

    def mark_unknown_outcome(
        self,
        dispatch_id: str,
        expected_version: int,
        reason_code: str,
    ) -> DispatchAttempt: ...

    def load_dispatch(self, dispatch_id: str) -> DispatchAttempt | None: ...

    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]: ...

    def assert_dispatch_allowed(self, game_id: str) -> None: ...


def require_durable_dispatch_repository(
    repository: object,
) -> DurableDispatchRepository:
    """只接受明确返回真值的 ``supports_durable_dispatch`` capability。"""

    supports = getattr(repository, "supports_durable_dispatch", None)
    if not callable(supports):
        raise DurableDispatchUnsupported(
            "repository does not support durable dispatch"
        )
    try:
        supported = supports()
    except Exception as exc:
        raise DurableDispatchUnsupported(
            "repository does not support durable dispatch"
        ) from exc
    if not supported:
        raise DurableDispatchUnsupported(
            "repository does not support durable dispatch"
        )
    return cast(DurableDispatchRepository, repository)


class RecoveryResolutionKind(StrEnum):
    """provider 重启恢复的封闭结果集合。"""

    FOUND = "found"
    REISSUED = "reissued"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"
    UNSAFE = "unsafe"


@dataclass(frozen=True, slots=True)
class RecoveryResolution:
    """resolver 对一个原始 dispatch attempt 的不可变恢复结论。"""

    kind: RecoveryResolutionKind
    result: DispatchResultRecord | None = None
    reason_code: str = ""


class DispatchResolver(Protocol):
    """查询或安全重发 provider 请求的解析器。"""

    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution: ...


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """一次游戏恢复扫描的不可变摘要。"""

    resolved: int = 0
    unknown: int = 0
    pending: int = 0
    errors: int = 0
    budget_consumption_required: bool = False
    barrier_open: bool = True


class DispatchReconciler:
    """按创建顺序协调重启后仍处于外部 I/O 状态的 dispatch。"""

    def __init__(
        self,
        repository: DurableDispatchRepository | object,
        resolver: DispatchResolver,
    ) -> None:
        self._repository = require_durable_dispatch_repository(repository)
        self._resolver = resolver

    def reconcile_game(self, game_id: str) -> RecoveryReport:
        """恢复单个游戏的未决 dispatch，并返回稳定计数摘要。"""

        try:
            attempts = self._repository.list_recoverable_dispatches(game_id)
            ordered = sorted(
                attempts,
                key=lambda attempt: (attempt.created_at, attempt.dispatch_id),
            )
        except Exception:  # noqa: BLE001 - storage failures become report errors
            return RecoveryReport(errors=1, barrier_open=False)

        resolved = 0
        unknown = 0
        pending = 0
        errors = 0
        budget_consumption_required = False

        for attempt in ordered:
            if not isinstance(attempt, DispatchAttempt):
                errors += 1
                continue

            if attempt.recovery_policy is DispatchRecoveryPolicy.AT_MOST_ONCE_UNKNOWN:
                try:
                    self._repository.mark_unknown_outcome(
                        attempt.dispatch_id,
                        expected_version=attempt.state_version,
                        reason_code="provider_not_idempotent",
                    )
                except Exception:  # noqa: BLE001 - one attempt must not abort recovery
                    errors += 1
                else:
                    unknown += 1
                    budget_consumption_required = True
                continue

            try:
                resolution = self._resolver.resolve(attempt)
            except Exception:  # noqa: BLE001 - resolver failures are report errors
                errors += 1
                continue

            if not isinstance(resolution, RecoveryResolution):
                errors += 1
                continue
            if not isinstance(resolution.kind, RecoveryResolutionKind):
                errors += 1
                continue
            if (
                resolution.result is not None
                and not isinstance(resolution.result, DispatchResultRecord)
            ):
                errors += 1
                continue

            kind = resolution.kind
            if kind is RecoveryResolutionKind.FOUND:
                if not self._result_matches_attempt(attempt, resolution.result):
                    errors += 1
                    continue
                try:
                    disposition = self._repository.record_result(
                        attempt.dispatch_id,
                        expected_version=attempt.state_version,
                        result=cast(DispatchResultRecord, resolution.result),
                    )
                except Exception:  # noqa: BLE001 - one attempt must not abort recovery
                    errors += 1
                else:
                    if disposition in {
                        DispatchResultDisposition.RECORDED,
                        DispatchResultDisposition.REPLAYED,
                    }:
                        resolved += 1
                    else:
                        errors += 1
                continue

            if kind is RecoveryResolutionKind.REISSUED:
                if resolution.result is None:
                    # 重新交付仍复用原 provider key；仓储状态不变，等待下一次扫描。
                    pending += 1
                    continue
                if not self._result_matches_attempt(attempt, resolution.result):
                    errors += 1
                    continue
                try:
                    disposition = self._repository.record_result(
                        attempt.dispatch_id,
                        expected_version=attempt.state_version,
                        result=resolution.result,
                    )
                except Exception:  # noqa: BLE001 - one attempt must not abort recovery
                    errors += 1
                else:
                    if disposition in {
                        DispatchResultDisposition.RECORDED,
                        DispatchResultDisposition.REPLAYED,
                    }:
                        resolved += 1
                    else:
                        errors += 1
                continue

            if kind in {
                RecoveryResolutionKind.PENDING,
                RecoveryResolutionKind.UNAVAILABLE,
            }:
                pending += 1
                continue

            if kind is RecoveryResolutionKind.UNSAFE:
                try:
                    self._repository.mark_unknown_outcome(
                        attempt.dispatch_id,
                        expected_version=attempt.state_version,
                        reason_code=resolution.reason_code or "unsafe_recovery",
                    )
                except Exception:  # noqa: BLE001 - one attempt must not abort recovery
                    errors += 1
                else:
                    unknown += 1
                    budget_consumption_required = True

        return RecoveryReport(
            resolved=resolved,
            unknown=unknown,
            pending=pending,
            errors=errors,
            budget_consumption_required=budget_consumption_required,
            # UNKNOWN_OUTCOME is terminal; only unresolved provider work or an
            # error leaves the recovery barrier closed in this report.
            barrier_open=pending == 0 and errors == 0,
        )

    @staticmethod
    def _result_matches_attempt(
        attempt: DispatchAttempt,
        result: DispatchResultRecord | None,
    ) -> bool:
        return (
            result is not None
            and result.dispatch_id == attempt.dispatch_id
            and result.request_hash == attempt.request_hash
            and result.lease_hash == attempt.lease_hash
        )


__all__ = [
    "DispatchIdempotencyConflict",
    "DispatchInvalidTransition",
    "DispatchLeaseMismatch",
    "DispatchNotFound",
    "DispatchReconciler",
    "DispatchRecoveryBlocked",
    "DispatchResolver",
    "DispatchResultConflict",
    "DispatchStateConflict",
    "DispatchTransactionError",
    "DurableDispatchError",
    "DurableDispatchRepository",
    "DurableDispatchUnsupported",
    "RecoveryReport",
    "RecoveryResolution",
    "RecoveryResolutionKind",
    "require_durable_dispatch_repository",
]
