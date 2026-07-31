# -*- coding: utf-8 -*-
"""
构建经过活动回合身份固定与最终复核的只读观察投影。

作者: Project contributors
创建日期: 2026-07-31

使用示例:
    >>> service.build_serial_public_observation("schedule-1", observed_at)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Never

from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurnStatus,
    ConflictClass,
)
from werewolf_agent.player_agents.observation.authority import (
    ObservationAuthorityReader,
    ObservationAuthoritySnapshot,
)
from werewolf_agent.player_agents.observation.contracts import (
    ObservationBundle,
    ObservationFrame,
    PlayerWorkspaceSnapshot,
    ProjectionIdentity,
)
from werewolf_agent.player_agents.observation.errors import (
    ActiveObservationConflict,
    ObservationProjectionError,
    ProjectionBuildFailed,
    ProjectionIdentityMismatch,
    ProjectionSourceChanged,
    ProjectionVisibilityRejected,
    RequiredProjectionUnavailable,
)
from werewolf_agent.player_agents.observation.workspace import WorkspaceProjector
from werewolf_agent.storage.autonomous_turns import (
    AutonomousTurnError,
    AutonomousTurnRepository,
    AutonomousTurnsUnsupported,
    require_autonomous_turn_repository,
)

_ALLOWED_OBSERVATION_STATUSES = frozenset({
    AgentTurnStatus.OPEN,
    AgentTurnStatus.OBSERVING,
})
_PINNED_SPEECH_ACTIONS = ("speech",)


def _is_aware(value: datetime) -> bool:
    """判断时间戳是否携带有效时区偏移。"""

    return value.tzinfo is not None and value.utcoffset() is not None


def _conflict() -> Never:
    """抛出不携带内部状态的活动观察冲突。"""

    raise ActiveObservationConflict()


def _validate_active_turn(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    observed_at: datetime,
) -> None:
    """验证一个调度与托管回合能否形成白天公开发言观察。"""

    slot_unavailable = False
    try:
        current_player_id = schedule.current_slot.player_id
    except (AttributeError, IndexError, ValueError):
        slot_unavailable = True
        current_player_id = ""
    if slot_unavailable:
        _conflict()
    turn = managed.turn
    if (
        schedule.status is not SerialPublicScheduleStatus.OPEN
        or schedule.active_turn_id is None
        or schedule.active_turn_id != turn.turn_id
        or managed.schedule_id != schedule.schedule_id
        or schedule.game_id != turn.game_id
        or schedule.window != turn.window
        or current_player_id != turn.player_id
        or schedule.window.task_type != "day_speech"
        or turn.task_type != "day_speech"
        or turn.phase != "day_discussion"
        or schedule.window.conflict_class is not ConflictClass.SERIAL_PUBLIC
        or turn.status not in _ALLOWED_OBSERVATION_STATUSES
        or schedule.window.legal_actions != _PINNED_SPEECH_ACTIONS
        or turn.revision.window_id != turn.window.window_id
        or turn.revision.window_version != turn.window.version
        or turn.revision.base_revision != turn.window.opened_revision
        or schedule.state_version < 0
        or managed.state_version < 0
        or not _is_aware(observed_at)
        or observed_at >= turn.window.deadline
    ):
        _conflict()


def prepare_observation_identity(
    first_schedule: SerialPublicSchedule,
    first_turn: ManagedAgentTurn,
    confirming_schedule: SerialPublicSchedule,
    observed_at: datetime,
) -> ProjectionIdentity:
    """从两次一致的调度读取中固定单次观察身份。"""

    _validate_active_turn(first_schedule, first_turn, observed_at)
    if confirming_schedule != first_schedule:
        _conflict()
    return ProjectionIdentity(
        game_id=first_turn.turn.game_id,
        player_id=first_turn.turn.player_id,
        schedule_id=first_schedule.schedule_id,
        turn_id=first_turn.turn.turn_id,
        schedule_state_version=first_schedule.state_version,
        turn_state_version=first_turn.state_version,
        window_id=first_turn.turn.window.window_id,
        window_version=first_turn.turn.window.version,
        base_game_revision=first_turn.turn.revision.base_revision,
        view_fingerprint=first_turn.turn.revision.view_fingerprint,
    )


def _require_authority_matches_turn(
    identity: ProjectionIdentity,
    managed: ManagedAgentTurn,
    authority: ObservationAuthoritySnapshot,
) -> None:
    """要求 Host 权威快照逐项匹配已固定的活动回合。"""

    if authority.persona is None or authority.role is None or authority.game is None:
        raise RequiredProjectionUnavailable()
    actual = authority.identity
    if (
        actual.game_id != identity.game_id
        or actual.player_id != identity.player_id
        or actual.schedule_id != identity.schedule_id
        or actual.turn_id != identity.turn_id
        or actual.window_id != identity.window_id
        or actual.window_version != identity.window_version
        or authority.role.role_id != managed.turn.role_id
        or authority.game.phase != managed.turn.phase
    ):
        raise ProjectionIdentityMismatch()
    if actual.view_fingerprint != identity.view_fingerprint:
        raise ProjectionVisibilityRejected()
    if (
        actual.schedule_state_version != identity.schedule_state_version
        or actual.turn_state_version != identity.turn_state_version
        or actual.base_game_revision != identity.base_game_revision
        or authority.legal_action_snapshot != managed.turn.window.legal_actions
        or authority.legal_target_snapshot != managed.turn.window.legal_target_ids
        or (
            *authority.game.authorized_private_fact_references,
            *authority.critical_private_fact_references,
        )
        != managed.turn.read_set
    ):
        raise ProjectionSourceChanged()

    source_identities = (
        authority.persona.source_identity,
        authority.role.source_identity,
        authority.game.source_identity,
        *(entry.source_identity for entry in authority.game.public_summary),
        *(
            entry.source_identity
            for entry in authority.commitment_records or ()
        ),
    )
    if any(source_identity != identity for source_identity in source_identities):
        raise ProjectionIdentityMismatch()


def assemble_observation_bundle(
    identity: ProjectionIdentity,
    managed: ManagedAgentTurn,
    authority: ObservationAuthoritySnapshot,
    workspace: PlayerWorkspaceSnapshot,
    observed_at: datetime,
) -> ObservationBundle:
    """把已验证权威、工作区与回合时限装配为不可变观察包。"""

    _require_authority_matches_turn(identity, managed, authority)
    if workspace.identity != identity:
        raise ProjectionIdentityMismatch()
    frame = ObservationFrame(
        identity=identity,
        task_kind="day_speech",
        actor_id=identity.player_id,
        role_id=managed.turn.role_id,
        phase="day_discussion",
        legal_action_snapshot=authority.legal_action_snapshot,
        legal_target_snapshot=authority.legal_target_snapshot,
        critical_private_fact_references=authority.critical_private_fact_references,
        bounded_public_summary=authority.bounded_public_summary,
        recent_commitment_references=authority.recent_commitment_references,
        document_manifest=workspace.manifest_entries,
        tool_manifest=authority.tool_manifest,
        workspace_revision=workspace.workspace_revision,
        workspace_hash=workspace.workspace_hash,
        deadline=managed.turn.window.deadline,
        observed_at=observed_at,
    )
    return ObservationBundle(frame=frame, workspace=workspace)


def require_unchanged_observation(
    identity: ProjectionIdentity,
    final_schedule: SerialPublicSchedule,
    final_turn: ManagedAgentTurn,
    observed_at: datetime,
    completed_at: datetime,
) -> None:
    """最终确认活动回合身份、版本与完成时间均未越界。"""

    _validate_active_turn(final_schedule, final_turn, observed_at)
    final_identity = ProjectionIdentity(
        game_id=final_turn.turn.game_id,
        player_id=final_turn.turn.player_id,
        schedule_id=final_schedule.schedule_id,
        turn_id=final_turn.turn.turn_id,
        schedule_state_version=final_schedule.state_version,
        turn_state_version=final_turn.state_version,
        window_id=final_turn.turn.window.window_id,
        window_version=final_turn.turn.window.version,
        base_game_revision=final_turn.turn.revision.base_revision,
        view_fingerprint=final_turn.turn.revision.view_fingerprint,
    )
    if (
        final_identity != identity
        or not _is_aware(completed_at)
        or completed_at < observed_at
        or completed_at >= final_turn.turn.window.deadline
    ):
        _conflict()


def _fresh_projection_error(
    error: ObservationProjectionError,
) -> ObservationProjectionError:
    """仅保留稳定投影错误类型与 code。"""

    return type(error)()


def _fresh_turn_error(error: AutonomousTurnError) -> AutonomousTurnError:
    """仅保留稳定仓储错误类型与 code。"""

    return type(error)(error.code)


class ObservationProjectionService:
    """在只读乐观围栏内构建一个活动白天发言观察。"""

    def __init__(
        self,
        repository: AutonomousTurnRepository | object,
        authority_reader: ObservationAuthorityReader,
        *,
        workspace_projector: WorkspaceProjector | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        capability_error = False
        try:
            self._repository = require_autonomous_turn_repository(repository)
        except AutonomousTurnsUnsupported:
            capability_error = True
        if capability_error:
            raise AutonomousTurnsUnsupported(
                AutonomousTurnsUnsupported.code,
            ) from None
        self._authority_reader = authority_reader
        self._workspace_projector = workspace_projector or WorkspaceProjector()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def build_serial_public_observation(
        self,
        schedule_id: str,
        observed_at: datetime,
    ) -> ObservationBundle:
        """构建观察，并在返回前重新确认活动回合仍完全相同。"""

        safe_error: ObservationProjectionError | AutonomousTurnError | None = None
        try:
            return self._build_serial_public_observation(
                schedule_id,
                observed_at,
            )
        except ObservationProjectionError as error:
            safe_error = _fresh_projection_error(error)
        except AutonomousTurnError as error:
            safe_error = _fresh_turn_error(error)
        except Exception:  # noqa: BLE001 - 所有实现细节必须在服务边界内净化。
            safe_error = ProjectionBuildFailed()
        raise safe_error from None

    def _build_serial_public_observation(
        self,
        schedule_id: str,
        observed_at: datetime,
    ) -> ObservationBundle:
        """执行两阶段乐观读取与纯投影装配。"""

        first_schedule = self._require_schedule(schedule_id)
        if first_schedule.active_turn_id is None:
            _conflict()
        first_turn = self._require_turn(first_schedule.active_turn_id)
        confirming_schedule = self._require_schedule(schedule_id)
        identity = prepare_observation_identity(
            first_schedule,
            first_turn,
            confirming_schedule,
            observed_at,
        )
        authority = self._authority_reader.read_observation_authority(
            identity,
            observed_at,
        )
        _require_authority_matches_turn(identity, first_turn, authority)
        workspace = self._workspace_projector.project(authority)
        bundle = assemble_observation_bundle(
            identity,
            first_turn,
            authority,
            workspace,
            observed_at,
        )
        final_schedule = self._require_schedule(schedule_id)
        final_turn = self._require_turn(identity.turn_id)
        if final_schedule != first_schedule or final_turn != first_turn:
            _conflict()
        completed_at = self._clock()
        require_unchanged_observation(
            identity,
            final_schedule,
            final_turn,
            observed_at,
            completed_at,
        )
        return bundle.model_copy(deep=True)

    def _require_schedule(self, schedule_id: str) -> SerialPublicSchedule:
        """读取指定调度，不把缺失状态暴露为部分观察。"""

        schedule = self._repository.load_serial_public_schedule(schedule_id)
        if schedule is None:
            _conflict()
        return schedule

    def _require_turn(self, turn_id: str) -> ManagedAgentTurn:
        """读取指定托管回合，不把缺失状态暴露为部分观察。"""

        managed = self._repository.load_managed_turn(turn_id)
        if managed is None:
            _conflict()
        return managed


__all__ = [
    "ObservationProjectionService",
    "assemble_observation_bundle",
    "prepare_observation_identity",
    "require_unchanged_observation",
]
