# -*- coding: utf-8 -*-
"""
验证自主玩家 durable dispatch、活动回合围栏与结果的严格契约。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-31
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.dispatch import (
    ActiveTurnDispatchFence,
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultOutcome,
    DispatchResultRecord,
    DispatchStatus,
)

HASH = "a" * 64


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
        "created_at": datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        "status": DispatchStatus.PENDING,
        "state_version": 0,
    }
    data.update(updates)
    return DispatchAttempt.model_validate(data)


def _fence(**updates: object) -> ActiveTurnDispatchFence:
    payload: dict[str, object] = {
        "schedule_id": "schedule-1",
        "schedule_state_version": 1,
        "turn_state_version": 4,
        "window_id": "speech-d1",
        "window_version": 1,
        "base_game_revision": 4,
    }
    payload.update(updates)
    return ActiveTurnDispatchFence.model_validate(payload)


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


def test_dispatch_attempt_is_strict_frozen_and_json_round_trips() -> None:
    attempt = _attempt()
    restored = DispatchAttempt.model_validate_json(attempt.model_dump_json())
    assert restored == attempt
    with pytest.raises(ValidationError):
        DispatchAttempt.model_validate({**attempt.model_dump(), "state_version": "0"})
    with pytest.raises(ValidationError):
        attempt.state_version = 1


def test_dispatch_attempt_rejects_naive_deadline_and_invalid_hash() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _attempt(deadline=datetime(2026, 7, 29, 12))  # noqa: DTZ001
    with pytest.raises(ValidationError):
        _attempt(request_hash="short")


def test_active_turn_fence_is_strict_frozen_and_json_round_trips() -> None:
    fence = _fence()
    assert ActiveTurnDispatchFence.model_validate_json(
        fence.model_dump_json(),
    ) == fence
    with pytest.raises(ValidationError):
        ActiveTurnDispatchFence.model_validate({
            **fence.model_dump(),
            "unexpected": True,
        })
    with pytest.raises((ValidationError, TypeError)):
        fence.turn_state_version = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "schedule_id",
        "schedule_state_version",
        "turn_state_version",
        "window_id",
        "window_version",
        "base_game_revision",
    ],
)
def test_active_turn_fence_requires_every_persisted_identity(
    field_name: str,
) -> None:
    payload = _fence().model_dump()
    payload.pop(field_name)
    with pytest.raises(ValidationError):
        ActiveTurnDispatchFence.model_validate(payload)


def test_dispatch_attempt_round_trips_optional_active_turn_fence() -> None:
    fenced = _attempt(active_turn_fence=_fence())
    restored = DispatchAttempt.model_validate_json(fenced.model_dump_json())
    assert restored == fenced
    assert restored.active_turn_fence == _fence()
    assert _attempt().active_turn_fence is None


def test_dispatch_result_payload_is_deeply_immutable() -> None:
    source_payload = {"nested": [{"safe": True}]}
    result = DispatchResultRecord(
        result_id="result-1",
        dispatch_id="dispatch-1",
        request_hash=HASH,
        lease_hash=HASH,
        result_hash=HASH,
        result_kind="model_response",
        outcome=DispatchResultOutcome.SUCCESS,
        payload=source_payload,
        recorded_at=datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )
    source_payload["nested"].append({"safe": False})
    with pytest.raises(TypeError):
        result.payload["nested"] = []  # type: ignore[index]
    with pytest.raises(TypeError):
        result.payload["nested"][0]["safe"] = False  # type: ignore[index]
    with pytest.raises((TypeError, AttributeError)):
        result.payload["nested"].append({"safe": False})  # type: ignore[union-attr]
    assert result.payload["nested"] == ({"safe": True},)
