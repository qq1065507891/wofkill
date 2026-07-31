# -*- coding: utf-8 -*-
"""
公开自主玩家观察投影的严格契约与稳定错误类型。

作者: Project contributors
创建日期: 2026-07-31
"""

from werewolf_agent.player_agents.observation.authority import (
    BoundedProjectionText,
    CommitmentProjectionSource,
    GameProjectionSource,
    ObservationAuthorityReader,
    ObservationAuthoritySnapshot,
    PersonaProjectionSource,
    PublicSummaryEntry,
    RoleAbilityProjectionSource,
    RoleProjectionSource,
)
from werewolf_agent.player_agents.observation.cache import (
    InMemoryProjectionCache,
    ProjectionCache,
    ProjectionCacheKey,
)
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
from werewolf_agent.player_agents.observation.rendering import (
    COMMITMENTS_RENDERER_VERSION,
    DOCUMENT_RENDERERS,
    GAME_RENDERER_VERSION,
    PLAYER_RENDERER_VERSION,
    ROLE_RENDERER_VERSION,
    ConservativeTokenEstimator,
    DocumentRenderer,
    TokenEstimator,
    render_commitments_document,
    render_game_document,
    render_player_document,
    render_role_document,
)
from werewolf_agent.player_agents.observation.service import (
    ObservationProjectionService,
    assemble_observation_bundle,
    prepare_observation_identity,
    require_unchanged_observation,
)
from werewolf_agent.player_agents.observation.workspace import (
    INDEX_RENDERER_VERSION,
    WorkspaceProjector,
)

__all__ = [
    "COMMITMENTS_RENDERER_VERSION",
    "DOCUMENT_RENDERERS",
    "GAME_RENDERER_VERSION",
    "INDEX_RENDERER_VERSION",
    "PLAYER_RENDERER_VERSION",
    "ROLE_RENDERER_VERSION",
    "ActiveObservationConflict",
    "BoundedObservationText",
    "BoundedProjectionText",
    "CommitmentProjectionSource",
    "ConservativeTokenEstimator",
    "DocumentRenderer",
    "GameProjectionSource",
    "InMemoryProjectionCache",
    "ManifestEntry",
    "ObservationAuthorityReader",
    "ObservationAuthoritySnapshot",
    "ObservationBundle",
    "ObservationFrame",
    "ObservationProjectionError",
    "ObservationProjectionService",
    "PersonaProjectionSource",
    "PlayerWorkspaceSnapshot",
    "ProjectedDocument",
    "ProjectionAvailability",
    "ProjectionBuildFailed",
    "ProjectionCache",
    "ProjectionCacheKey",
    "ProjectionIdentity",
    "ProjectionIdentityMismatch",
    "ProjectionIntegrityFailed",
    "ProjectionRenderFailed",
    "ProjectionSourceChanged",
    "ProjectionSourceReference",
    "ProjectionUnavailableReason",
    "ProjectionVisibilityClass",
    "ProjectionVisibilityRejected",
    "PublicSummaryEntry",
    "RequiredProjectionUnavailable",
    "RoleAbilityProjectionSource",
    "RoleProjectionSource",
    "TokenEstimator",
    "WorkspaceProjector",
    "WorkspaceSection",
    "assemble_observation_bundle",
    "prepare_observation_identity",
    "render_commitments_document",
    "render_game_document",
    "render_player_document",
    "render_role_document",
    "require_unchanged_observation",
]
