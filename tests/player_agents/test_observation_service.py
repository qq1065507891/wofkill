# -*- coding: utf-8 -*-
"""
验证活动串行公开回合的乐观固定观察构建服务。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import TypeVar

import pytest

from werewolf_agent.core.models import GameState
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
    SerialPublicSlot,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
)
from werewolf_agent.player_agents.observation import (
    ActiveObservationConflict,
    GameProjectionSource,
    InMemoryProjectionCache,
    ObservationAuthorityReader,
    ObservationAuthoritySnapshot,
    ObservationProjectionError,
    ObservationProjectionService,
    PersonaProjectionSource,
    PlayerWorkspaceSnapshot,
    ProjectionBuildFailed,
    ProjectionIdentity,
    ProjectionRenderFailed,
    ProjectionSourceReference,
    RoleProjectionSource,
    WorkspaceProjector,
    prepare_observation_identity,
)
from werewolf_agent.storage.autonomous_turns import (
    AutonomousTurnsUnsupported,
    ScheduleNotFound,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)
DEADLINE = NOW + timedelta(hours=1)
T = TypeVar("T")


def _unchecked_replace(value: T, **updates: object) -> T:
    """构造用于验证服务二次防御的已损坏边界对象。"""

    copied = deepcopy(value)
    for field_name, replacement in updates.items():
        object.__setattr__(copied, field_name, replacement)
    return copied


def _source(
    kind: str,
    record_id: str,
    revision: int,
    content_hash: str,
) -> ProjectionSourceReference:
    return ProjectionSourceReference(
        record_kind=kind,
        record_id=record_id,
        record_revision=revision,
        content_hash=content_hash,
    )


def _memory_active_turn(
    repository: InMemoryGameRepository | None = None,
) -> tuple[
    InMemoryGameRepository,
    SerialPublicSchedule,
    ManagedAgentTurn,
]:
    repository = repository or InMemoryGameRepository()
    repository.save_game(GameState(game_id="game-1"))
    window = LegalActionWindow(
        window_id="speech-d1",
        version=1,
        game_id="game-1",
        task_type="day_speech",
        conflict_class=ConflictClass.SERIAL_PUBLIC,
        participant_ids=("p01", "p02"),
        legal_actions=("speech",),
        legal_target_ids=("p01", "p02"),
        opened_revision=4,
        deadline=DEADLINE,
    )
    schedule = repository.create_serial_public_schedule(SerialPublicSchedule(
        schedule_id="schedule-1",
        game_id="game-1",
        window=window,
        slots=(
            SerialPublicSlot(ordinal=0, player_id="p01"),
            SerialPublicSlot(ordinal=1, player_id="p02"),
        ),
        next_slot_ordinal=0,
        active_turn_id=None,
        status=SerialPublicScheduleStatus.OPEN,
        state_version=0,
        created_at=NOW,
        updated_at=NOW,
    ))
    managed = repository.admit_serial_public_turn(
        schedule.schedule_id,
        schedule.state_version,
        TurnAdmission(
            turn_id="turn-1",
            player_id="p01",
            role_id="villager",
            phase="day_discussion",
            revision=RevisionContext(
                base_revision=4,
                window_id=window.window_id,
                window_version=window.version,
                view_fingerprint=HASH_A,
            ),
            read_set=(
                ReadReference(
                    record_id="public-4",
                    revision=4,
                    content_hash=HASH_B,
                ),
            ),
            model_lease_hash=HASH_C,
            budget=TurnBudget(model_steps=8, tool_calls=0, repairs=1),
            idempotency_key="turn-1:submit",
        ),
    )
    active_schedule = repository.load_serial_public_schedule(schedule.schedule_id)
    assert active_schedule is not None
    return repository, active_schedule, managed


def _identity_for(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> ProjectionIdentity:
    return ProjectionIdentity(
        game_id=managed.turn.game_id,
        player_id=managed.turn.player_id,
        schedule_id=schedule.schedule_id,
        turn_id=managed.turn.turn_id,
        schedule_state_version=schedule.state_version,
        turn_state_version=managed.state_version,
        window_id=managed.turn.window.window_id,
        window_version=managed.turn.window.version,
        base_game_revision=managed.turn.revision.base_revision,
        view_fingerprint=managed.turn.revision.view_fingerprint,
    )


def _authority_snapshot_for(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> ObservationAuthoritySnapshot:
    identity = _identity_for(schedule, managed)
    return ObservationAuthoritySnapshot(
        identity=identity,
        persona=PersonaProjectionSource(
            profile_id="persona-1",
            profile_version="v1",
            display_name="清醒村民",
            personality_summary="先验证证据，再表达判断。",
            risk_appetite="中等",
            source_identity=identity,
            source_reference=_source("persona", "persona-1", 1, HASH_A),
        ),
        role=RoleProjectionSource(
            role_id=managed.turn.role_id,
            faction_id="good",
            role_summary="没有夜间技能。",
            source_identity=identity,
            source_reference=_source("role", "p01-role", 4, HASH_B),
        ),
        game=GameProjectionSource(
            day=1,
            phase=managed.turn.phase,
            living_player_ids=("p01", "p02"),
            authorized_private_fact_references=managed.turn.read_set,
            source_identity=identity,
            source_references=(_source("game", "game-1", 4, HASH_C),),
        ),
        commitment_records=None,
        legal_action_snapshot=managed.turn.window.legal_actions,
        legal_target_snapshot=managed.turn.window.legal_target_ids,
        critical_private_fact_references=(),
        bounded_public_summary=("白天讨论继续。",),
        recent_commitment_references=(),
    )


class FakeAuthorityReader:
    def __init__(self, snapshot: ObservationAuthoritySnapshot) -> None:
        self._snapshot = snapshot

    def read_observation_authority(
        self,
        identity: ProjectionIdentity,
        observed_at: datetime,
    ) -> ObservationAuthoritySnapshot:
        return self._snapshot.model_copy(deep=True)


def _service(
    repository: InMemoryGameRepository,
    snapshot: ObservationAuthoritySnapshot,
    *,
    clock: Callable[[], datetime] = lambda: NOW,
) -> ObservationProjectionService:
    reader: ObservationAuthorityReader = FakeAuthorityReader(snapshot)
    return ObservationProjectionService(repository, reader, clock=clock)


def test_service_builds_one_active_day_speech_observation() -> None:
    repository, schedule, managed = _memory_active_turn()
    service = _service(repository, _authority_snapshot_for(schedule, managed))
    bundle = service.build_serial_public_observation(schedule.schedule_id, NOW)
    assert bundle.frame.identity.schedule_id == schedule.schedule_id
    assert bundle.frame.identity.turn_id == managed.turn.turn_id
    assert bundle.frame.identity.player_id == managed.turn.player_id
    assert bundle.frame.legal_action_snapshot == managed.turn.window.legal_actions
    assert bundle.frame.workspace_hash == bundle.workspace.workspace_hash
    assert repository.load_serial_public_schedule(schedule.schedule_id) == schedule
    assert repository.load_managed_turn(managed.turn.turn_id) == managed
    assert not hasattr(bundle.frame, "dispatch_id")


def test_service_rejects_repository_without_turn_capability() -> None:
    _, schedule, managed = _memory_active_turn()
    reader = FakeAuthorityReader(_authority_snapshot_for(schedule, managed))
    with pytest.raises(AutonomousTurnsUnsupported):
        ObservationProjectionService(object(), reader)


def _capture_with_mutation(
    mutation: str,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn, SerialPublicSchedule, datetime]:
    _, schedule, managed = _memory_active_turn()
    confirming = schedule
    observed_at = NOW
    turn = managed.turn
    if mutation == "closed_schedule":
        schedule = schedule.model_copy(update={
            "status": SerialPublicScheduleStatus.CLOSED,
            "active_turn_id": None,
        })
    elif mutation == "missing_active_turn":
        schedule = schedule.model_copy(update={"active_turn_id": None})
    elif mutation == "other_schedule":
        managed = managed.model_copy(update={"schedule_id": "schedule-2"})
    elif mutation == "other_turn":
        schedule = schedule.model_copy(update={"active_turn_id": "turn-2"})
    elif mutation == "other_game":
        managed = _unchecked_replace(
            managed,
            turn=_unchecked_replace(turn, game_id="game-2"),
        )
    elif mutation == "other_player":
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"player_id": "p02"}),
        })
    elif mutation == "other_phase":
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"phase": "night"}),
        })
    elif mutation == "other_task":
        window = turn.window.model_copy(update={"task_type": "vote"})
        schedule = _unchecked_replace(schedule, window=window)
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"task_type": "vote", "window": window}),
        })
    elif mutation == "other_conflict_class":
        window = turn.window.model_copy(
            update={"conflict_class": ConflictClass.SERIAL_PRIVATE},
        )
        schedule = _unchecked_replace(schedule, window=window)
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"window": window}),
        })
    elif mutation == "other_window_id":
        window = turn.window.model_copy(update={"window_id": "speech-d2"})
        revision = turn.revision.model_copy(update={"window_id": "speech-d2"})
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"window": window, "revision": revision}),
        })
    elif mutation == "other_window_version":
        window = turn.window.model_copy(update={"version": 2})
        revision = turn.revision.model_copy(update={"window_version": 2})
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"window": window, "revision": revision}),
        })
    elif mutation == "stale_base_revision":
        revision = turn.revision.model_copy(update={"base_revision": 3})
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"revision": revision}),
        })
    elif mutation == "other_legal_actions":
        window = turn.window.model_copy(update={"legal_actions": ("vote",)})
        schedule = schedule.model_copy(update={"window": window})
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"window": window}),
        })
    elif mutation == "other_legal_targets":
        window = turn.window.model_copy(update={"legal_target_ids": ("p02",)})
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"window": window}),
        })
    elif mutation == "other_deadline":
        window = turn.window.model_copy(
            update={"deadline": DEADLINE + timedelta(minutes=1)},
        )
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"window": window}),
        })
    elif mutation == "negative_schedule_version":
        schedule = _unchecked_replace(schedule, state_version=-1)
    elif mutation == "negative_turn_version":
        managed = _unchecked_replace(managed, state_version=-1)
    elif mutation == "thinking":
        managed = managed.model_copy(update={
            "turn": turn.model_copy(update={"status": AgentTurnStatus.THINKING}),
        })
    elif mutation == "schedule_changed":
        confirming = schedule.model_copy(
            update={"state_version": schedule.state_version + 1},
        )
    elif mutation == "naive_observed_at":
        observed_at = NOW.replace(tzinfo=None)
    elif mutation == "expired":
        observed_at = DEADLINE
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    if mutation != "schedule_changed":
        confirming = schedule
    return schedule, managed, confirming, observed_at


@pytest.mark.parametrize(
    "mutation",
    [
        "closed_schedule",
        "missing_active_turn",
        "other_schedule",
        "other_turn",
        "other_game",
        "other_player",
        "other_phase",
        "other_task",
        "other_conflict_class",
        "other_window_id",
        "other_window_version",
        "stale_base_revision",
        "other_legal_actions",
        "other_legal_targets",
        "other_deadline",
        "negative_schedule_version",
        "negative_turn_version",
        "thinking",
        "schedule_changed",
        "naive_observed_at",
        "expired",
    ],
)
def test_prepare_identity_rejects_non_pinned_active_turn(mutation: str) -> None:
    schedule, managed, confirming, observed_at = _capture_with_mutation(mutation)
    with pytest.raises(ActiveObservationConflict):
        prepare_observation_identity(
            schedule,
            managed,
            confirming,
            observed_at,
        )


def _mutated_authority(
    snapshot: ObservationAuthoritySnapshot,
    mutation: str,
) -> ObservationAuthoritySnapshot:
    identity = snapshot.identity
    identity_mutations: dict[str, tuple[str, object]] = {
        "other_game": ("game_id", "game-2"),
        "other_player": ("player_id", "p02"),
        "other_schedule": ("schedule_id", "schedule-2"),
        "other_turn": ("turn_id", "turn-2"),
        "other_window_id": ("window_id", "speech-d2"),
        "other_window_version": ("window_version", 2),
        "other_schedule_version": ("schedule_state_version", 99),
        "other_turn_version": ("turn_state_version", 99),
    }
    if mutation in identity_mutations:
        field_name, replacement = identity_mutations[mutation]
        return _unchecked_replace(
            snapshot,
            identity=identity.model_copy(update={field_name: replacement}),
        )
    if mutation == "stale_revision":
        return _unchecked_replace(
            snapshot,
            identity=identity.model_copy(update={"base_game_revision": 3}),
        )
    if mutation == "changed_view":
        return _unchecked_replace(
            snapshot,
            identity=identity.model_copy(update={"view_fingerprint": HASH_C}),
        )
    if mutation == "other_role":
        return snapshot.model_copy(update={
            "role": snapshot.role.model_copy(update={"role_id": "seer"}),
        })
    if mutation == "other_phase":
        return snapshot.model_copy(update={
            "game": snapshot.game.model_copy(update={"phase": "night"}),
        })
    if mutation == "other_legal_actions":
        return snapshot.model_copy(update={"legal_action_snapshot": ("vote",)})
    if mutation == "other_legal_targets":
        return snapshot.model_copy(update={"legal_target_snapshot": ("p02",)})
    if mutation == "other_reads":
        return snapshot.model_copy(update={
            "game": snapshot.game.model_copy(
                update={"authorized_private_fact_references": ()},
            ),
        })
    if mutation == "missing_player":
        return _unchecked_replace(snapshot, persona=None)
    if mutation == "source_identity":
        other = identity.model_copy(update={"turn_id": "turn-2"})
        persona = snapshot.persona.model_copy(update={"source_identity": other})
        return _unchecked_replace(snapshot, persona=persona)
    raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("other_game", "projection_identity_mismatch"),
        ("other_player", "projection_identity_mismatch"),
        ("other_schedule", "projection_identity_mismatch"),
        ("other_turn", "projection_identity_mismatch"),
        ("other_window_id", "projection_identity_mismatch"),
        ("other_window_version", "projection_identity_mismatch"),
        ("other_schedule_version", "projection_source_changed"),
        ("other_turn_version", "projection_source_changed"),
        ("stale_revision", "projection_source_changed"),
        ("changed_view", "projection_visibility_rejected"),
        ("other_role", "projection_identity_mismatch"),
        ("other_phase", "projection_identity_mismatch"),
        ("other_legal_actions", "projection_source_changed"),
        ("other_legal_targets", "projection_source_changed"),
        ("other_reads", "projection_source_changed"),
        ("missing_player", "required_projection_unavailable"),
        ("source_identity", "projection_identity_mismatch"),
    ],
)
def test_service_fails_closed_without_partial_bundle(
    mutation: str,
    expected_code: str,
) -> None:
    repository, schedule, managed = _memory_active_turn()
    snapshot = _mutated_authority(
        _authority_snapshot_for(schedule, managed),
        mutation,
    )
    with pytest.raises(ObservationProjectionError) as exc_info:
        _service(repository, snapshot).build_serial_public_observation(
            schedule.schedule_id,
            NOW,
        )
    assert exc_info.value.code == expected_code
    assert not hasattr(exc_info.value, "partial_bundle")


class InterleavingAuthorityReader(FakeAuthorityReader):
    """读取权威期间只通过仓储公开 API 改变一次活动回合。"""

    def __init__(
        self,
        repository: InMemoryGameRepository,
        schedule: SerialPublicSchedule,
        managed: ManagedAgentTurn,
        snapshot: ObservationAuthoritySnapshot,
        interleaving: str,
    ) -> None:
        super().__init__(snapshot)
        self._repository = repository
        self._schedule = schedule
        self._managed = managed
        self._interleaving = interleaving

    def read_observation_authority(
        self,
        identity: ProjectionIdentity,
        observed_at: datetime,
    ) -> ObservationAuthoritySnapshot:
        if self._interleaving == "observing_then_thinking":
            observing = self._repository.transition_active_turn(
                self._managed.turn.turn_id,
                self._managed.state_version,
                AgentTurnStatus.OBSERVING,
            )
            self._repository.transition_active_turn(
                observing.turn.turn_id,
                observing.state_version,
                AgentTurnStatus.THINKING,
            )
        else:
            disposition = (
                TerminalDisposition.REPLACE
                if self._interleaving == "replace"
                else TerminalDisposition.CLOSE
            )
            terminal_status = (
                AgentTurnStatus.EXPIRED
                if self._interleaving == "expire"
                else AgentTurnStatus.CANCELLED
            )
            changed = self._repository.finish_active_turn(
                self._schedule.schedule_id,
                self._schedule.state_version,
                self._managed.turn.turn_id,
                self._managed.state_version,
                terminal_status,
                disposition,
                "interleaved_observation_change",
            )
            if self._interleaving == "replace":
                self._repository.admit_serial_public_turn(
                    changed.schedule_id,
                    changed.state_version,
                    TurnAdmission(
                        turn_id="turn-2",
                        player_id=self._managed.turn.player_id,
                        role_id=self._managed.turn.role_id,
                        phase=self._managed.turn.phase,
                        revision=self._managed.turn.revision,
                        read_set=self._managed.turn.read_set,
                        model_lease_hash=HASH_B,
                        budget=self._managed.turn.budget,
                        idempotency_key="turn-2:submit",
                    ),
                )
        return super().read_observation_authority(identity, observed_at)


@pytest.mark.parametrize(
    "interleaving",
    ["cancel", "expire", "observing_then_thinking", "replace"],
)
def test_service_rejects_turn_changed_during_projection(
    interleaving: str,
) -> None:
    repository, schedule, managed = _memory_active_turn()
    snapshot = _authority_snapshot_for(schedule, managed)
    reader = InterleavingAuthorityReader(
        repository,
        schedule,
        managed,
        snapshot,
        interleaving,
    )
    service = ObservationProjectionService(
        repository,
        reader,
        workspace_projector=WorkspaceProjector(cache=InMemoryProjectionCache()),
        clock=lambda: NOW,
    )
    with pytest.raises(ActiveObservationConflict):
        service.build_serial_public_observation(schedule.schedule_id, NOW)


def test_cached_workspace_cannot_bypass_final_recheck() -> None:
    repository, schedule, managed = _memory_active_turn()
    snapshot = _authority_snapshot_for(schedule, managed)
    cache = InMemoryProjectionCache()
    first_service = ObservationProjectionService(
        repository,
        FakeAuthorityReader(snapshot),
        workspace_projector=WorkspaceProjector(cache=cache),
        clock=lambda: NOW,
    )
    first_service.build_serial_public_observation(schedule.schedule_id, NOW)

    interleaving_reader = InterleavingAuthorityReader(
        repository,
        schedule,
        managed,
        snapshot,
        "cancel",
    )
    second_service = ObservationProjectionService(
        repository,
        interleaving_reader,
        workspace_projector=WorkspaceProjector(cache=cache),
        clock=lambda: NOW,
    )
    with pytest.raises(ActiveObservationConflict):
        second_service.build_serial_public_observation(schedule.schedule_id, NOW)


@pytest.mark.parametrize(
    "completed_at",
    [NOW.replace(tzinfo=None), NOW - timedelta(microseconds=1), DEADLINE],
)
def test_service_rejects_invalid_completion_clock(completed_at: datetime) -> None:
    repository, schedule, managed = _memory_active_turn()
    service = _service(
        repository,
        _authority_snapshot_for(schedule, managed),
        clock=lambda: completed_at,
    )
    with pytest.raises(ActiveObservationConflict):
        service.build_serial_public_observation(schedule.schedule_id, NOW)


class FinalLeaseMutationRepository(InMemoryGameRepository):
    """仅在最终读取时模拟未递增版本的错误后端。"""

    def __init__(self) -> None:
        super().__init__()
        self._turn_reads = 0

    def load_managed_turn(self, turn_id: str) -> ManagedAgentTurn | None:
        managed = super().load_managed_turn(turn_id)
        self._turn_reads += 1
        if managed is not None and self._turn_reads == 2:
            turn = managed.turn.model_copy(update={"model_lease_hash": HASH_A})
            return managed.model_copy(update={"turn": turn})
        return managed


def test_service_rechecks_exact_turn_lease_not_only_version() -> None:
    repository, schedule, managed = _memory_active_turn(
        FinalLeaseMutationRepository(),
    )
    service = _service(repository, _authority_snapshot_for(schedule, managed))
    with pytest.raises(ActiveObservationConflict):
        service.build_serial_public_observation(schedule.schedule_id, NOW)


def test_pure_capture_helper_does_not_leak_validation_context() -> None:
    schedule, managed, _, _ = _capture_with_mutation("closed_schedule")
    with pytest.raises(ActiveObservationConflict) as exc_info:
        prepare_observation_identity(schedule, managed, schedule, NOW)
    error = exc_info.value
    assert error.__cause__ is None
    assert error.__context__ is None


class RaisingAuthorityReader:
    def read_observation_authority(
        self,
        identity: ProjectionIdentity,
        observed_at: datetime,
    ) -> ObservationAuthoritySnapshot:
        raise RuntimeError("private-reader-marker")


class ChainedProjectionReader:
    def read_observation_authority(
        self,
        identity: ProjectionIdentity,
        observed_at: datetime,
    ) -> ObservationAuthoritySnapshot:
        try:
            raise RuntimeError("private-reader-marker")
        except RuntimeError as error:
            raise ProjectionRenderFailed() from error


@pytest.mark.parametrize(
    ("reader_type", "expected_error"),
    [
        (RaisingAuthorityReader, ProjectionBuildFailed),
        (ChainedProjectionReader, ProjectionRenderFailed),
    ],
)
def test_service_sanitizes_reader_failures(
    reader_type: type[RaisingAuthorityReader | ChainedProjectionReader],
    expected_error: type[ObservationProjectionError],
) -> None:
    repository, schedule, _ = _memory_active_turn()
    reader: ObservationAuthorityReader = reader_type()
    service = ObservationProjectionService(repository, reader, clock=lambda: NOW)
    with pytest.raises(expected_error) as exc_info:
        service.build_serial_public_observation(schedule.schedule_id, NOW)
    error = exc_info.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "private-reader-marker" not in "".join(traceback.format_exception(error))


class RaisingLoadRepository(InMemoryGameRepository):
    def load_serial_public_schedule(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None:
        raise RuntimeError("private-repository-marker")


class ChainedExpectedRepository(InMemoryGameRepository):
    def load_serial_public_schedule(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None:
        try:
            raise RuntimeError("private-repository-marker")
        except RuntimeError as error:
            raise ScheduleNotFound("private schedule details") from error


class RaisingWorkspaceProjector:
    def project(
        self,
        snapshot: ObservationAuthoritySnapshot,
    ) -> PlayerWorkspaceSnapshot:
        raise RuntimeError("private-projector-marker")


def test_service_sanitizes_unexpected_repository_failure() -> None:
    repository = RaisingLoadRepository()
    reader: ObservationAuthorityReader = RaisingAuthorityReader()
    service = ObservationProjectionService(repository, reader, clock=lambda: NOW)
    with pytest.raises(ProjectionBuildFailed) as exc_info:
        service.build_serial_public_observation("schedule-1", NOW)
    error = exc_info.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "private-repository-marker" not in "".join(
        traceback.format_exception(error),
    )


def test_service_preserves_only_safe_repository_error_class_and_code() -> None:
    repository = ChainedExpectedRepository()
    reader: ObservationAuthorityReader = RaisingAuthorityReader()
    service = ObservationProjectionService(repository, reader, clock=lambda: NOW)
    with pytest.raises(ScheduleNotFound) as exc_info:
        service.build_serial_public_observation("schedule-1", NOW)
    error = exc_info.value
    assert error.code == "schedule_not_found"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "private-repository-marker" not in "".join(
        traceback.format_exception(error),
    )


def test_service_sanitizes_unexpected_projector_failure() -> None:
    repository, schedule, managed = _memory_active_turn()
    reader: ObservationAuthorityReader = FakeAuthorityReader(
        _authority_snapshot_for(schedule, managed),
    )
    service = ObservationProjectionService(
        repository,
        reader,
        workspace_projector=RaisingWorkspaceProjector(),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    with pytest.raises(ProjectionBuildFailed) as exc_info:
        service.build_serial_public_observation(schedule.schedule_id, NOW)
    error = exc_info.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "private-projector-marker" not in "".join(
        traceback.format_exception(error),
    )
