# -*- coding: utf-8 -*-
"""
验证 GameEvent V2 元数据盖章、序列化和 V1 只读兼容。

作者: Project contributors
创建日期: 2026-07-15
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.event_metadata import (
    deserialize_game_event,
    serialize_game_event,
    stamp_new_events,
)
from werewolf_agent.runtime.game_runner_execution import GameRunnerExecutionMixin


def test_stamp_new_events_assigns_stable_v2_metadata() -> None:
    fixed_now = datetime(2026, 7, 15, 8, 30, tzinfo=timezone.utc)
    before: list[GameEvent] = []
    after = [GameEvent(type="enter_night", payload={"night_number": 1})]

    stamped = stamp_new_events("g1", before, after, now=fixed_now)

    assert stamped[0].event_id == "g1:e000000"
    assert stamped[0].sequence_number == 0
    assert stamped[0].occurred_at == fixed_now
    assert stamped[0].occurred_at.tzinfo is not None
    assert stamped[0].game_id == "g1"
    assert stamped[0].schema_version == "2"
    assert stamped[0].visibility is EventVisibility.PUBLIC


def test_stamp_new_events_promotes_legacy_visibility_without_payload_duplicate() -> None:
    event = GameEvent(
        type="seer_check",
        payload={"target_id": "p02", "visibility": "seer_private"},
    )

    stamped = stamp_new_events("g1", [], [event])

    assert stamped[0].visibility is EventVisibility.SEER_PRIVATE
    assert stamped[0].payload == {"target_id": "p02"}


def test_stamp_new_events_preserves_existing_v2_metadata() -> None:
    fixed_now = datetime(2026, 7, 15, 8, 30, tzinfo=timezone.utc)
    existing = GameEvent(
        type="enter_night",
        event_id="g1:e000004",
        sequence_number=4,
        occurred_at=fixed_now,
        game_id="g1",
        visibility=EventVisibility.PUBLIC,
        schema_version="2",
    )
    unstamped = GameEvent(type="day_announce")

    stamped = stamp_new_events("g1", [existing], [existing, unstamped])

    assert stamped[0] is existing
    assert stamped[1].event_id == "g1:e000005"
    assert stamped[1].sequence_number == 5


def test_legacy_game_event_remains_v1_and_reads_payload_visibility() -> None:
    event = GameEvent(
        type="wolf_discussion",
        payload={"visibility": "werewolf_team_only"},
    )

    assert event.event_id is None
    assert event.sequence_number is None
    assert event.occurred_at is None
    assert event.game_id is None
    assert event.schema_version is None
    assert event.visibility is None
    assert event_visibility(event) is EventVisibility.WEREWOLF_TEAM_ONLY


def test_v2_top_level_visibility_is_authoritative() -> None:
    event = GameEvent(
        type="judge_broadcast",
        payload={"visibility": "public"},
        visibility=EventVisibility.MODERATOR_ONLY,
        schema_version="2",
    )

    assert event_visibility(event) is EventVisibility.MODERATOR_ONLY


def test_event_serializer_round_trips_datetime_and_enum() -> None:
    fixed_now = datetime(2026, 7, 15, 8, 30, tzinfo=timezone.utc)
    event = stamp_new_events(
        "g1",
        [],
        [GameEvent(type="enter_night", trace_id="trace-1")],
        now=fixed_now,
    )[0]

    serialized = serialize_game_event(event)
    restored = deserialize_game_event(serialized)

    assert serialized["occurred_at"] == "2026-07-15T08:30:00+00:00"
    assert serialized["visibility"] == "public"
    assert restored == event


def test_v2_serializer_removes_conflicting_legacy_payload_visibility() -> None:
    event = GameEvent(
        type="speech",
        payload={"text": "private", "visibility": "public"},
        visibility=EventVisibility.ACTOR_PRIVATE,
        event_id="g1:e000000",
        sequence_number=0,
        occurred_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        game_id="g1",
        schema_version="2",
    )

    serialized = serialize_game_event(event)

    assert serialized["visibility"] == "actor_private"
    assert serialized["payload"] == {"text": "private"}


def test_v1_serializer_preserves_legacy_payload_visibility_for_read_compatibility() -> None:
    event = GameEvent(
        type="speech",
        payload={"text": "legacy", "visibility": "moderator_only"},
    )

    serialized = serialize_game_event(event)

    assert serialized["visibility"] is None
    assert serialized["payload"]["visibility"] == "moderator_only"


def test_game_event_rejects_naive_occurred_at_in_memory() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        GameEvent(type="invalid", occurred_at=datetime(2026, 7, 15))


def test_runner_stamps_only_events_added_by_current_node() -> None:
    class CognitionManager:
        def update_from_events(self, state: object) -> None:
            self.state = state

    runner = GameRunnerExecutionMixin()
    runner._state = GameState(
        game_id="g1",
        events=[stamp_new_events("g1", [], [GameEvent(type="existing")])[0]],
    )
    runner._cognition_state_manager = CognitionManager()
    existing = runner._state.events[0]
    next_state = GameState(
        game_id="g1",
        events=[existing, GameEvent(type="enter_night")],
    )

    runner._process_chunk({"enter_night": {"game_state": next_state}})

    assert runner._state.events[0] is existing
    assert runner._state.events[1].event_id == "g1:e000001"
    assert runner._state.events[1].schema_version == "2"


def test_memory_repository_preserves_complete_v2_dataclass() -> None:
    from werewolf_agent.storage.memory_store import InMemoryGameRepository

    event = stamp_new_events("g1", [], [GameEvent(type="enter_night")])[0]
    repository = InMemoryGameRepository()

    repository.append_events("g1", [event])

    assert repository.load_events("g1")[0] is event


def test_v2_private_visibility_is_used_by_public_share_and_seer_strategy() -> None:
    from werewolf_agent.api.routes.game_public_share import _event_is_public_for_share
    from werewolf_agent.runtime.strategy.seer import public_seer_claimants

    event = GameEvent(
        type="speech",
        payload={"speaker": "p01", "text": "我是预言家"},
        visibility=EventVisibility.ACTOR_PRIVATE,
        schema_version="2",
    )

    assert _event_is_public_for_share(event) is False
    assert public_seer_claimants(GameState(events=[event])) == set()


def test_runner_stamps_hitl_events_outside_process_chunk() -> None:
    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig

    class FakeHitl:
        is_paused = True
        _pending_command = object()

        def send_command(self, raw: str) -> None:
            self.raw = raw

        def handle_command(self, command: object, state: GameState) -> dict:
            return {"game_state": state, "response": "OK"}

        def flush_events(self) -> list[GameEvent]:
            return [GameEvent(type="hitl_audit")]

    runner = GameRunner(GameRunnerConfig(seed=42, use_agent_registry=False))
    existing = stamp_new_events(
        runner.game_id,
        [],
        [GameEvent(type="existing")],
    )[0]
    runner._state = GameState(game_id=runner.game_id, events=[existing])
    runner._hitl_interface = FakeHitl()

    runner.send_command("status")

    assert runner.state.events[0] is existing
    assert runner.state.events[1].event_id == f"{runner.game_id}:e000001"
    assert runner.state.events[1].sequence_number == 1
    assert runner.state.events[1].schema_version == "2"


def test_runner_stamps_terminal_reflection_audit_continuously() -> None:
    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
    from werewolf_agent.storage.memory_store import InMemoryGameRepository

    repository = InMemoryGameRepository()
    runner = GameRunner(GameRunnerConfig(
        seed=43,
        use_agent_registry=False,
        repository=repository,
    ))
    existing = stamp_new_events(
        runner.game_id,
        [],
        [GameEvent(type="existing")],
    )[0]
    runner._state = GameState(game_id=runner.game_id, events=[existing])

    runner._append_reflection_persistence_audit([], upstream_complete=False)

    audit = runner.state.events[-1]
    assert audit.event_id == f"{runner.game_id}:e000001"
    assert audit.sequence_number == 1
    assert audit.visibility is EventVisibility.MODERATOR_ONLY
    assert "visibility" not in audit.payload
    assert audit.schema_version == "2"


def test_runner_rollback_update_preserves_reflection_audit_v2_metadata() -> None:
    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
    from werewolf_agent.storage.memory_store import InMemoryGameRepository

    runner = GameRunner(GameRunnerConfig(
        seed=44,
        use_agent_registry=False,
        repository=InMemoryGameRepository(),
    ))
    runner._append_reflection_persistence_audit([], upstream_complete=False)
    before = runner.state.events[-1]

    runner._set_latest_reflection_rollback_status(False)

    after = runner.state.events[-1]
    assert after.event_id == before.event_id
    assert after.sequence_number == before.sequence_number
    assert after.occurred_at == before.occurred_at
    assert after.visibility == before.visibility
    assert after.schema_version == "2"
    assert after.payload["rollback_complete"] is False
