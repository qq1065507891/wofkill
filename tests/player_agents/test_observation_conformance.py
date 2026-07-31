# -*- coding: utf-8 -*-
"""
验证观察投影在 Memory 与 SQLite 后端上的一致性、重启确定性和无文件副作用。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
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
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import (
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
)
from werewolf_agent.player_agents.observation import (
    GameProjectionSource,
    InMemoryProjectionCache,
    ObservationAuthorityReader,
    ObservationAuthoritySnapshot,
    ObservationProjectionError,
    ObservationProjectionService,
    PersonaProjectionSource,
    ProjectionIdentity,
    ProjectionSourceReference,
    RoleProjectionSource,
    WorkspaceProjector,
)
from werewolf_agent.storage.autonomous_turns import AutonomousTurnRepository
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)
DEADLINE = datetime(2036, 7, 31, 12, tzinfo=timezone.utc)
T = TypeVar("T")

RepositoryFactory = Callable[
    [Path],
    tuple[AutonomousTurnRepository, SerialPublicSchedule, ManagedAgentTurn],
]
ConcreteRepository = InMemoryGameRepository | SqliteGameRepository


def _unchecked_replace(value: T, **updates: object) -> T:
    """构造绕过模型构造校验的敌对边界输入。"""

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


def _seed_active_turn(
    repository: ConcreteRepository,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
    """用同一组字面量在任一真实后端创建活动公开发言回合。"""

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
    created = repository.create_serial_public_schedule(SerialPublicSchedule(
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
        created.schedule_id,
        created.state_version,
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
            read_set=(ReadReference(
                record_id="public-4",
                revision=4,
                content_hash=HASH_B,
            ),),
            model_lease_hash=HASH_C,
            budget=TurnBudget(model_steps=8, tool_calls=0, repairs=1),
            idempotency_key="turn-1:submit",
        ),
    )
    schedule = repository.load_serial_public_schedule(created.schedule_id)
    assert schedule is not None
    return schedule, managed


def _memory_factory(
    _tmp_path: Path,
) -> tuple[AutonomousTurnRepository, SerialPublicSchedule, ManagedAgentTurn]:
    repository = InMemoryGameRepository()
    schedule, managed = _seed_active_turn(repository)
    return repository, schedule, managed


def _sqlite_factory(
    tmp_path: Path,
) -> tuple[AutonomousTurnRepository, SerialPublicSchedule, ManagedAgentTurn]:
    repository = SqliteGameRepository(str(tmp_path / "observation.db"))
    schedule, managed = _seed_active_turn(repository)
    return repository, schedule, managed


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
    *,
    living_player_ids: tuple[str, ...] | list[str] = ("p01", "p02"),
    game_source_references: tuple[ProjectionSourceReference, ...]
    | list[ProjectionSourceReference]
    | None = None,
    legal_actions: tuple[str, ...] | list[str] | None = None,
    legal_targets: tuple[str, ...] | list[str] | None = None,
    public_summary: tuple[str, ...] | list[str] = ("白天讨论继续。",),
) -> ObservationAuthoritySnapshot:
    """从活动回合构造只含当前查看者授权事实的进程内权威。"""

    identity = _identity_for(schedule, managed)
    sources = game_source_references or [_source("game", "game-1", 4, HASH_C)]
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
            living_player_ids=living_player_ids,
            authorized_private_fact_references=managed.turn.read_set,
            source_identity=identity,
            source_references=sources,
        ),
        commitment_records=None,
        legal_action_snapshot=legal_actions or list(managed.turn.window.legal_actions),
        legal_target_snapshot=legal_targets or list(
            managed.turn.window.legal_target_ids,
        ),
        critical_private_fact_references=(),
        bounded_public_summary=public_summary,
        recent_commitment_references=(),
    )


class _AuthorityReader:
    def __init__(self, snapshot: ObservationAuthoritySnapshot) -> None:
        self._snapshot = snapshot

    def read_observation_authority(
        self,
        identity: ProjectionIdentity,
        observed_at: datetime,
    ) -> ObservationAuthoritySnapshot:
        return self._snapshot.model_copy(deep=True)


def _service(
    repository: AutonomousTurnRepository,
    authority: ObservationAuthoritySnapshot,
    *,
    projector: WorkspaceProjector | None = None,
    completed_at: datetime,
) -> ObservationProjectionService:
    reader: ObservationAuthorityReader = _AuthorityReader(authority)
    return ObservationProjectionService(
        repository,
        reader,
        workspace_projector=projector,
        clock=lambda: completed_at,
    )


def _close(repository: AutonomousTurnRepository) -> None:
    if isinstance(repository, SqliteGameRepository):
        repository.close()


@pytest.mark.parametrize("repository_factory", [_memory_factory, _sqlite_factory])
def test_observation_backend_conformance(
    repository_factory: RepositoryFactory,
    tmp_path: Path,
) -> None:
    repository, schedule, managed = repository_factory(tmp_path)
    try:
        authority = _authority_snapshot_for(schedule, managed)
        first = _service(
            repository,
            authority,
            completed_at=managed.updated_at,
        ).build_serial_public_observation(schedule.schedule_id, managed.updated_at)
        second = _service(
            repository,
            authority,
            completed_at=managed.updated_at,
        ).build_serial_public_observation(schedule.schedule_id, managed.updated_at)
        assert first == second
        assert first.workspace.workspace_hash == second.workspace.workspace_hash
        assert first.frame.identity.turn_id == managed.turn.turn_id
    finally:
        _close(repository)


def _error_authority(
    scenario: str,
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> tuple[ObservationAuthoritySnapshot, datetime]:
    authority = _authority_snapshot_for(schedule, managed)
    observed_at = managed.updated_at
    if scenario == "cross_player":
        identity = authority.identity.model_copy(update={"player_id": "p02"})
        authority = _unchecked_replace(authority, identity=identity)
    elif scenario == "stale_revision":
        identity = authority.identity.model_copy(update={"base_game_revision": 3})
        authority = _unchecked_replace(authority, identity=identity)
    elif scenario == "changed_view":
        identity = authority.identity.model_copy(update={"view_fingerprint": HASH_C})
        authority = _unchecked_replace(authority, identity=identity)
    elif scenario == "expired_deadline":
        observed_at = managed.turn.window.deadline
    elif scenario == "required_source_absence":
        authority = _unchecked_replace(authority, persona=None)
    else:
        raise AssertionError(f"unknown scenario: {scenario}")
    return authority, observed_at


@pytest.mark.parametrize("repository_factory", [_memory_factory, _sqlite_factory])
@pytest.mark.parametrize(
    ("scenario", "expected_code"),
    [
        ("cross_player", "projection_identity_mismatch"),
        ("stale_revision", "projection_source_changed"),
        ("changed_view", "projection_visibility_rejected"),
        ("expired_deadline", "active_observation_conflict"),
        ("required_source_absence", "required_projection_unavailable"),
    ],
)
def test_observation_backend_error_code_conformance(
    repository_factory: RepositoryFactory,
    scenario: str,
    expected_code: str,
    tmp_path: Path,
) -> None:
    repository, schedule, managed = repository_factory(tmp_path)
    try:
        authority, observed_at = _error_authority(scenario, schedule, managed)
        service = _service(
            repository,
            authority,
            completed_at=managed.updated_at,
        )
        with pytest.raises(ObservationProjectionError) as exc_info:
            service.build_serial_public_observation(
                schedule.schedule_id,
                observed_at,
            )
        assert exc_info.value.code == expected_code
        assert not hasattr(exc_info.value, "partial_bundle")
    finally:
        _close(repository)


def test_sqlite_restart_rebuilds_byte_identical_workspace(tmp_path: Path) -> None:
    repository, schedule, managed = _sqlite_factory(tmp_path)
    assert isinstance(repository, SqliteGameRepository)
    try:
        authority = _authority_snapshot_for(schedule, managed)
        first = _service(
            repository,
            authority,
            completed_at=managed.updated_at,
        ).build_serial_public_observation(schedule.schedule_id, managed.updated_at)
    finally:
        repository.close()

    reopened = SqliteGameRepository(str(tmp_path / "observation.db"))
    try:
        reopened_schedule = reopened.load_serial_public_schedule(schedule.schedule_id)
        reopened_managed = reopened.load_managed_turn(managed.turn.turn_id)
        assert reopened_schedule is not None
        assert reopened_managed is not None
        rebuilt_authority = _authority_snapshot_for(
            reopened_schedule,
            reopened_managed,
        )
        second = _service(
            reopened,
            rebuilt_authority,
            completed_at=reopened_managed.updated_at,
        ).build_serial_public_observation(
            reopened_schedule.schedule_id,
            reopened_managed.updated_at,
        )
        assert tuple(
            document.model_dump_json().encode("utf-8")
            for document in first.workspace.documents
        ) == tuple(
            document.model_dump_json().encode("utf-8")
            for document in second.workspace.documents
        )
        assert first.workspace.workspace_revision == second.workspace.workspace_revision
        assert first.workspace.workspace_hash == second.workspace.workspace_hash
    finally:
        reopened.close()


def test_sqlite_projection_defends_against_caller_owned_source_mutation(
    tmp_path: Path,
) -> None:
    repository, schedule, managed = _sqlite_factory(tmp_path)
    assert isinstance(repository, SqliteGameRepository)
    living_player_ids = ["p01", "p02"]
    game_sources = [_source("game", "game-1", 4, HASH_C)]
    legal_actions = ["speech"]
    legal_targets = ["p01", "p02"]
    public_summary = ["白天讨论继续。"]
    authority = _authority_snapshot_for(
        schedule,
        managed,
        living_player_ids=living_player_ids,
        game_source_references=game_sources,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        public_summary=public_summary,
    )
    projector = WorkspaceProjector(cache=InMemoryProjectionCache())
    try:
        first = _service(
            repository,
            authority,
            projector=projector,
            completed_at=managed.updated_at,
        ).build_serial_public_observation(schedule.schedule_id, managed.updated_at)

        living_player_ids[:] = ["p99"]
        game_sources.clear()
        legal_actions[:] = ["vote"]
        legal_targets.clear()
        public_summary[:] = ["已被调用方篡改。"]

        second = _service(
            repository,
            authority,
            projector=projector,
            completed_at=managed.updated_at,
        ).build_serial_public_observation(schedule.schedule_id, managed.updated_at)
        assert second == first
        assert second.workspace.documents == first.workspace.documents
    finally:
        repository.close()


def test_projection_creates_no_physical_workspace_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, schedule, managed = _memory_factory(tmp_path)
    projection_cwd = tmp_path / "empty-projection-cwd"
    projection_cwd.mkdir()
    monkeypatch.chdir(projection_cwd)

    authority = _authority_snapshot_for(schedule, managed)
    _service(
        repository,
        authority,
        completed_at=managed.updated_at,
    ).build_serial_public_observation(schedule.schedule_id, managed.updated_at)

    assert tuple(projection_cwd.iterdir()) == ()
