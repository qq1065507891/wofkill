# -*- coding: utf-8 -*-
"""
提供仅进程内使用且防御性复制的观察投影文档缓存。

作者: Project contributors
创建日期: 2026-07-31

使用示例:
    >>> isinstance(InMemoryProjectionCache(), ProjectionCache)
    True
"""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Protocol, runtime_checkable

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)
from werewolf_agent.player_agents.observation.contracts import (
    ProjectedDocument,
    WorkspaceSection,
)
from werewolf_agent.player_agents.observation.errors import (
    ProjectionIdentityMismatch,
    ProjectionIntegrityFailed,
)


class ProjectionCacheKey(StrictFrozenModel):
    """唯一绑定一份派生文档的缓存键。"""

    game_id: NonEmptyId
    player_id: NonEmptyId
    view_fingerprint: ContentHash
    workspace_revision: ContentHash
    section_id: WorkspaceSection
    renderer_version: NonEmptyId
    estimator_version: NonEmptyId


@runtime_checkable
class ProjectionCache(Protocol):
    """可选缓存的最小接口，不暴露存储或失效控制。"""

    def get(self, key: ProjectionCacheKey) -> ProjectedDocument | None:
        """返回独立副本，未命中时返回 ``None``。"""

    def put(self, key: ProjectionCacheKey, document: ProjectedDocument) -> None:
        """保存独立副本。"""


def _validate_key_document_binding(
    key: ProjectionCacheKey,
    document: ProjectedDocument,
) -> None:
    """拒绝跨游戏、玩家或视图的缓存内容。"""

    identity = document.identity
    if (
        identity.game_id != key.game_id
        or identity.player_id != key.player_id
        or identity.view_fingerprint != key.view_fingerprint
    ):
        raise ProjectionIdentityMismatch()
    if (
        document.section_id != key.section_id
        or document.renderer_version != key.renderer_version
        or document.estimator_version != key.estimator_version
    ):
        raise ProjectionIntegrityFailed()


class InMemoryProjectionCache:
    """RLock 保护的进程内缓存；不提供任何持久化路径。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._documents: dict[ProjectionCacheKey, ProjectedDocument] = {}

    def get(self, key: ProjectionCacheKey) -> ProjectedDocument | None:
        """返回深复制结果，防止调用方改变缓存持有值。"""

        with self._lock:
            document = self._documents.get(key)
            if document is None:
                return None
            _validate_key_document_binding(key, document)
            return deepcopy(document)

    def put(self, key: ProjectionCacheKey, document: ProjectedDocument) -> None:
        """先验证绑定，再保存深复制值。"""

        _validate_key_document_binding(key, document)
        with self._lock:
            self._documents[key] = deepcopy(document)


__all__ = [
    "InMemoryProjectionCache",
    "ProjectionCache",
    "ProjectionCacheKey",
]
