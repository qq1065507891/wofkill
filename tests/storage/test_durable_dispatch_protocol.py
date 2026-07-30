# -*- coding: utf-8 -*-
"""
验证 durable dispatch 能力守卫、恢复解析器与重启协调器契约。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-29
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.core.models import GameState
from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultDisposition,
    DispatchResultOutcome,
    DispatchResultRecord,
    DispatchStatus,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchIdempotencyConflict,
    DispatchInvalidTransition,
    DispatchLeaseMismatch,
    DispatchNotFound,
    DispatchReconciler,
    DispatchRecoveryBlocked,
    DispatchResultConflict,
    DispatchStateConflict,
    DispatchTransactionError,
    DurableDispatchUnsupported,
    RecoveryResolution,
    RecoveryResolutionKind,
    require_durable_dispatch_repository,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository

HASH = "a" * 64
NOW = datetime(2026, 7, 29, 11, tzinfo=timezone.utc)


def _attempt(**updates: object) -> DispatchAttempt:
    data: dict[str, object] = {
        "dispatch_id": "dispatch-1",
        "game_id": "game-1",
        "turn_id": "turn-1",
        "actor_id": "p01",
        "operation_kind": DispatchOperationKind.MODEL,
        "executor_id": "mock-provider",
        "provider_idempotency_key": "provider-key-1",
        "recovery_policy": DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE,
        "request_hash": HASH,
        "lease_hash": HASH,
        "view_fingerprint": HASH,
        "deadline": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
        "created_at": NOW,
        "updated_at": NOW,
        "status": DispatchStatus.PENDING,
        "state_version": 0,
    }
    data.update(updates)
    return DispatchAttempt.model_validate(data)


def _result(**updates: object) -> DispatchResultRecord:
    data: dict[str, object] = {
        "result_id": "result-1",
        "dispatch_id": "dispatch-1",
        "request_hash": HASH,
        "lease_hash": HASH,
        "result_hash": HASH,
        "result_kind": "model_response",
        "outcome": DispatchResultOutcome.SUCCESS,
        "payload": {"accepted": True},
        "recorded_at": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    }
    data.update(updates)
    return DispatchResultRecord.model_validate(data)


class InMemoryDispatchFixture:
    """测试用的最小内存仓储，模拟 CAS 与结果绑定。"""

    def __init__(self, attempts: list[DispatchAttempt]) -> None:
        self._attempts = {item.dispatch_id: item for item in attempts}
        self.results: dict[str, DispatchResultRecord] = {}
        self.unknown_reasons: list[str] = []

    def supports_durable_dispatch(self) -> bool:
        return True

    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt:
        if attempt.dispatch_id in self._attempts:
            raise DispatchIdempotencyConflict(attempt.dispatch_id)
        self._attempts[attempt.dispatch_id] = attempt
        return attempt

    def mark_dispatching(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt:
        return self._transition(
            dispatch_id,
            expected_version,
            {DispatchStatus.PENDING},
            DispatchStatus.DISPATCHING,
        )

    def mark_dispatched(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt:
        return self._transition(
            dispatch_id,
            expected_version,
            {DispatchStatus.DISPATCHING},
            DispatchStatus.DISPATCHED,
        )

    def record_result(
        self,
        dispatch_id: str,
        expected_version: int,
        result: DispatchResultRecord,
    ) -> DispatchResultDisposition:
        attempt = self._attempts.get(dispatch_id)
        if attempt is None:
            raise DispatchNotFound(dispatch_id)
        if attempt.state_version != expected_version:
            raise DispatchStateConflict(dispatch_id)
        if result.dispatch_id != dispatch_id:
            raise DispatchResultConflict(dispatch_id)
        if result.request_hash != attempt.request_hash:
            raise DispatchResultConflict(dispatch_id)
        if result.lease_hash != attempt.lease_hash:
            raise DispatchLeaseMismatch(dispatch_id)
        prior = self.results.get(dispatch_id)
        if prior is not None:
            if prior == result:
                return DispatchResultDisposition.REPLAYED
            raise DispatchResultConflict(dispatch_id)
        if attempt.status is not DispatchStatus.DISPATCHED:
            raise DispatchInvalidTransition(dispatch_id)
        self.results[dispatch_id] = result
        self._attempts[dispatch_id] = attempt.model_copy(
            update={
                "status": DispatchStatus.RESULT_RECORDED,
                "state_version": attempt.state_version + 1,
            }
        )
        return DispatchResultDisposition.RECORDED

    def cancel_dispatch(
        self,
        dispatch_id: str,
        expected_version: int,
        reason_code: str,
    ) -> DispatchAttempt:
        return self._transition(
            dispatch_id,
            expected_version,
            {DispatchStatus.PENDING, DispatchStatus.DISPATCHING},
            DispatchStatus.CANCELLED,
            reason_code,
        )

    def mark_unknown_outcome(
        self,
        dispatch_id: str,
        expected_version: int,
        reason_code: str,
    ) -> DispatchAttempt:
        result = self._transition(
            dispatch_id,
            expected_version,
            {DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED},
            DispatchStatus.UNKNOWN_OUTCOME,
            reason_code,
        )
        self.unknown_reasons.append(reason_code)
        return result

    def load_dispatch(self, dispatch_id: str) -> DispatchAttempt | None:
        return self._attempts.get(dispatch_id)

    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]:
        return [
            attempt
            for attempt in self._attempts.values()
            if attempt.game_id == game_id
            and attempt.status
            in {DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED}
        ]

    def list_dispatches_for_turn(
        self,
        game_id: str,
        turn_id: str,
    ) -> list[DispatchAttempt]:
        attempts = [
            attempt
            for attempt in self._attempts.values()
            if attempt.game_id == game_id and attempt.turn_id == turn_id
        ]
        attempts.sort(key=lambda item: (item.created_at, item.dispatch_id))
        return [attempt.model_copy(deep=True) for attempt in attempts]

    def assert_dispatch_allowed(self, game_id: str) -> None:
        if self.list_recoverable_dispatches(game_id):
            raise DispatchRecoveryBlocked(game_id)

    def _transition(
        self,
        dispatch_id: str,
        expected_version: int,
        allowed: set[DispatchStatus],
        target: DispatchStatus,
        reason_code: str | None = None,
    ) -> DispatchAttempt:
        attempt = self._attempts.get(dispatch_id)
        if attempt is None:
            raise DispatchNotFound(dispatch_id)
        if attempt.state_version != expected_version:
            raise DispatchStateConflict(dispatch_id)
        if attempt.status not in allowed:
            raise DispatchInvalidTransition(dispatch_id)
        updated = attempt.model_copy(
            update={
                "status": target,
                "state_version": attempt.state_version + 1,
                "reason_code": reason_code,
            }
        )
        self._attempts[dispatch_id] = updated
        return updated


class NeverCalledResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        raise AssertionError(f"resolver must not run for {attempt.dispatch_id}")


class FoundResolver:
    def __init__(self, result: DispatchResultRecord) -> None:
        self.result = result
        self.seen_keys: list[str] = []

    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        self.seen_keys.append(attempt.provider_idempotency_key)
        return RecoveryResolution(
            kind=RecoveryResolutionKind.FOUND,
            result=self.result,
        )


class PendingResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        return RecoveryResolution(kind=RecoveryResolutionKind.PENDING)


class ReissuedResolver:
    def __init__(self, result: DispatchResultRecord) -> None:
        self.result = result
        self.seen_keys: list[str] = []

    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        self.seen_keys.append(attempt.provider_idempotency_key)
        return RecoveryResolution(
            kind=RecoveryResolutionKind.REISSUED,
            result=self.result,
        )


class UnavailableResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        return RecoveryResolution(kind=RecoveryResolutionKind.UNAVAILABLE)


class UnsafeResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        return RecoveryResolution(
            kind=RecoveryResolutionKind.UNSAFE,
            reason_code="unsafe_provider_binding",
        )


def test_capability_guard_requires_explicit_support() -> None:
    class PlainObject:
        def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt:
            return attempt

        def commit_turn(self, request: object) -> object:
            return request

    with pytest.raises(DurableDispatchUnsupported):
        require_durable_dispatch_repository(PlainObject())


def test_storage_errors_expose_stable_codes() -> None:
    errors = (
        DurableDispatchUnsupported,
        DispatchNotFound,
        DispatchStateConflict,
        DispatchInvalidTransition,
        DispatchIdempotencyConflict,
        DispatchLeaseMismatch,
        DispatchResultConflict,
        DispatchRecoveryBlocked,
        DispatchTransactionError,
    )
    assert [error.code for error in errors] == [
        "durable_dispatch_unsupported",
        "dispatch_not_found",
        "dispatch_state_conflict",
        "dispatch_invalid_transition",
        "dispatch_idempotency_conflict",
        "dispatch_lease_mismatch",
        "dispatch_result_conflict",
        "dispatch_recovery_blocked",
        "dispatch_transaction_error",
    ]


def test_recovery_resolution_is_frozen_and_kind_is_closed() -> None:
    resolution = RecoveryResolution(kind=RecoveryResolutionKind.PENDING)
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        resolution.kind = RecoveryResolutionKind.UNSAFE  # type: ignore[misc]
    assert {kind.value for kind in RecoveryResolutionKind} == {
        "found",
        "reissued",
        "pending",
        "unavailable",
        "unsafe",
    }


def test_dispatches_for_turn_fixture_is_deterministic_and_scoped() -> None:
    store = InMemoryDispatchFixture(
        [
            _attempt(
                dispatch_id="dispatch-z",
                game_id="game-1",
                turn_id="turn-2",
                created_at=datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
            ),
            _attempt(
                dispatch_id="dispatch-b",
                game_id="game-1",
                turn_id="turn-1",
                created_at=datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
                status=DispatchStatus.CANCELLED,
                state_version=1,
            ),
            _attempt(
                dispatch_id="dispatch-a",
                game_id="game-1",
                turn_id="turn-1",
                created_at=datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
                status=DispatchStatus.RESULT_RECORDED,
                state_version=3,
            ),
            _attempt(
                dispatch_id="dispatch-other-game",
                game_id="game-2",
                turn_id="turn-1",
            ),
        ]
    )

    listed = store.list_dispatches_for_turn("game-1", "turn-1")

    assert [item.dispatch_id for item in listed] == ["dispatch-a", "dispatch-b"]
    assert {item.status for item in listed} == {
        DispatchStatus.CANCELLED,
        DispatchStatus.RESULT_RECORDED,
    }
    assert listed[0] is not store.load_dispatch("dispatch-a")


def test_reconciler_marks_non_idempotent_attempt_unknown() -> None:
    store = InMemoryDispatchFixture(
        [
            _attempt(
                recovery_policy=DispatchRecoveryPolicy.AT_MOST_ONCE_UNKNOWN,
                status=DispatchStatus.DISPATCHED,
                state_version=2,
            )
        ]
    )
    report = DispatchReconciler(
        store,
        resolver=NeverCalledResolver(),
    ).reconcile_game("game-1")
    assert report.unknown == 1
    assert store.load_dispatch("dispatch-1").status is DispatchStatus.UNKNOWN_OUTCOME  # type: ignore[union-attr]
    assert report.barrier_open is True
    assert store.unknown_reasons == ["provider_not_idempotent"]


def test_reconciler_records_found_idempotent_result_without_new_dispatch_id() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHED, state_version=2)]
    )
    resolver = FoundResolver(_result())
    report = DispatchReconciler(store, resolver=resolver).reconcile_game("game-1")
    assert report.resolved == 1
    assert store.load_dispatch("dispatch-1").status is DispatchStatus.RESULT_RECORDED  # type: ignore[union-attr]
    assert resolver.seen_keys == ["provider-key-1"]
    assert set(store._attempts) == {"dispatch-1"}


def test_reconciler_reissued_result_keeps_attempt_dispatched_and_reuses_key() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHED, state_version=2)]
    )
    resolver = ReissuedResolver(_result())

    report = DispatchReconciler(store, resolver=resolver).reconcile_game("game-1")

    assert report.resolved == 0
    assert report.pending == 1
    assert report.errors == 0
    assert report.barrier_open is False
    attempt = store.load_dispatch("dispatch-1")
    assert attempt is not None
    assert attempt.status is DispatchStatus.DISPATCHED
    assert attempt.state_version == 2
    assert attempt.provider_idempotency_key == "provider-key-1"
    assert resolver.seen_keys == ["provider-key-1"]
    assert store.results == {}


def test_reconciler_reissued_dispatching_promotes_to_dispatched() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHING, state_version=1)]
    )
    resolver = ReissuedResolver(_result())

    report = DispatchReconciler(store, resolver=resolver).reconcile_game("game-1")

    assert report.pending == 1
    assert report.resolved == 0
    assert report.errors == 0
    attempt = store.load_dispatch("dispatch-1")
    assert attempt is not None
    assert attempt.status is DispatchStatus.DISPATCHED
    assert attempt.state_version == 2
    assert attempt.provider_idempotency_key == "provider-key-1"
    assert store.results == {}


def test_reconciler_leaves_pending_provider_and_keeps_barrier_closed() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHING, state_version=1)]
    )
    report = DispatchReconciler(store, resolver=PendingResolver()).reconcile_game("game-1")
    assert report.pending == 1
    assert report.barrier_open is False
    with pytest.raises(DispatchRecoveryBlocked):
        store.assert_dispatch_allowed("game-1")


def test_reconciler_leaves_unavailable_provider_pending() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHING, state_version=1)]
    )

    report = DispatchReconciler(
        store,
        resolver=UnavailableResolver(),
    ).reconcile_game("game-1")

    assert report.pending == 1
    assert report.barrier_open is False
    attempt = store.load_dispatch("dispatch-1")
    assert attempt is not None
    assert attempt.status is DispatchStatus.DISPATCHING
    assert attempt.state_version == 1


def test_reconciler_marks_unsafe_resolution_unknown() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHED, state_version=2)]
    )

    report = DispatchReconciler(
        store,
        resolver=UnsafeResolver(),
    ).reconcile_game("game-1")

    assert report.unknown == 1
    assert report.budget_consumption_required is True
    assert report.barrier_open is True
    attempt = store.load_dispatch("dispatch-1")
    assert attempt is not None
    assert attempt.status is DispatchStatus.UNKNOWN_OUTCOME
    assert attempt.reason_code == "unsafe_provider_binding"


def test_reconciler_rejects_found_result_bound_to_different_dispatch() -> None:
    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHED, state_version=2)]
    )
    resolver = FoundResolver(_result(dispatch_id="dispatch-other"))

    report = DispatchReconciler(store, resolver=resolver).reconcile_game("game-1")

    assert report.errors == 1
    assert report.resolved == 0
    assert store.results == {}
    attempt = store.load_dispatch("dispatch-1")
    assert attempt is not None
    assert attempt.status is DispatchStatus.DISPATCHED
    assert attempt.state_version == 2


def test_reconciler_passes_immutable_attempt_to_resolver() -> None:
    seen: list[DispatchAttempt] = []

    class InspectingResolver:
        def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
            seen.append(attempt)
            with pytest.raises((AttributeError, TypeError, ValidationError)):
                attempt.status = DispatchStatus.UNKNOWN_OUTCOME  # type: ignore[misc]
            return RecoveryResolution(kind=RecoveryResolutionKind.UNAVAILABLE)

    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHED, state_version=1)]
    )
    DispatchReconciler(store, resolver=InspectingResolver()).reconcile_game("game-1")
    assert seen[0] == _attempt(status=DispatchStatus.DISPATCHED, state_version=1)


def test_reconciler_rejects_invalid_resolver_output_without_mutating_attempt() -> None:
    class InvalidResolver:
        def resolve(self, attempt: DispatchAttempt) -> object:
            return object()

    store = InMemoryDispatchFixture(
        [_attempt(status=DispatchStatus.DISPATCHED, state_version=1)]
    )
    report = DispatchReconciler(store, resolver=InvalidResolver()).reconcile_game("game-1")
    assert report.errors == 1
    assert store.load_dispatch("dispatch-1").status is DispatchStatus.DISPATCHED  # type: ignore[union-attr]


def test_reconciler_found_dispatching_records_the_same_dispatch_id() -> None:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="game-1"))
    repository.create_dispatch(_attempt())
    repository.mark_dispatching("dispatch-1", expected_version=0)

    class Resolver:
        def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
            assert attempt.dispatch_id == "dispatch-1"
            return RecoveryResolution(
                kind=RecoveryResolutionKind.FOUND,
                result=_result(),
            )

    report = DispatchReconciler(repository, resolver=Resolver()).reconcile_game(
        "game-1",
    )

    assert report.resolved == 1
    assert report.barrier_open is True
    loaded = repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.RESULT_RECORDED
    assert loaded.state_version == 3


def test_reconciler_counts_dispatching_found_promotion_failure() -> None:
    class FailingPromotionStore(InMemoryDispatchFixture):
        def __init__(self) -> None:
            super().__init__([_attempt(status=DispatchStatus.DISPATCHING, state_version=1)])
            self.calls: list[tuple[str, int]] = []

        def mark_dispatched(self, dispatch_id: str, expected_version: int) -> DispatchAttempt:
            self.calls.append((dispatch_id, expected_version))
            raise DispatchStateConflict(dispatch_id)

    store = FailingPromotionStore()
    report = DispatchReconciler(
        store,
        resolver=FoundResolver(_result()),
    ).reconcile_game("game-1")

    assert report.errors == 1
    assert report.resolved == 0
    assert report.barrier_open is False
    assert store.calls == [("dispatch-1", 1)]
    assert store.results == {}
    attempt = store.load_dispatch("dispatch-1")
    assert attempt is not None
    assert attempt.status is DispatchStatus.DISPATCHING
    assert attempt.state_version == 1
