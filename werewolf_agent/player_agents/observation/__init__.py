# -*- coding: utf-8 -*-
"""
公开自主玩家观察投影的严格契约与稳定错误类型。

作者: Project contributors
创建日期: 2026-07-31
"""

from werewolf_agent.player_agents.observation.contracts import (
    BoundedObservationText,
    ManifestEntry,
    ObservationBundle,
    ObservationFrame,
    PlayerWorkspaceSnapshot,
    ProjectedDocument,
    ProjectionAvailability,
    ProjectionIdentity,
    ProjectionSourceReference,
    ProjectionUnavailableReason,
    ProjectionVisibilityClass,
    WorkspaceSection,
)
from werewolf_agent.player_agents.observation.errors import (
    ActiveObservationConflict,
    ObservationProjectionError,
    ProjectionBuildFailed,
    ProjectionIdentityMismatch,
    ProjectionIntegrityFailed,
    ProjectionRenderFailed,
    ProjectionSourceChanged,
    ProjectionVisibilityRejected,
    RequiredProjectionUnavailable,
)

__all__ = [
    "ActiveObservationConflict",
    "BoundedObservationText",
    "ManifestEntry",
    "ObservationBundle",
    "ObservationFrame",
    "ObservationProjectionError",
    "PlayerWorkspaceSnapshot",
    "ProjectedDocument",
    "ProjectionAvailability",
    "ProjectionBuildFailed",
    "ProjectionIdentity",
    "ProjectionIdentityMismatch",
    "ProjectionIntegrityFailed",
    "ProjectionRenderFailed",
    "ProjectionSourceChanged",
    "ProjectionSourceReference",
    "ProjectionUnavailableReason",
    "ProjectionVisibilityClass",
    "ProjectionVisibilityRejected",
    "RequiredProjectionUnavailable",
    "WorkspaceSection",
]
