# -*- coding: utf-8 -*-
"""
定义观察投影流程可安全暴露的稳定错误类型。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations


class ObservationProjectionError(RuntimeError):
    """观察投影错误的稳定基类，不接受可能含私密信息的载荷。"""

    code = "observation_projection_error"

    def __init__(self) -> None:
        super().__init__(self.code)


class ActiveObservationConflict(ObservationProjectionError):
    """表示同一玩家存在冲突的活动观察。"""

    code = "active_observation_conflict"


class RequiredProjectionUnavailable(ObservationProjectionError):
    """表示必需投影不可用。"""

    code = "required_projection_unavailable"


class ProjectionIdentityMismatch(ObservationProjectionError):
    """表示投影身份与绑定上下文不一致。"""

    code = "projection_identity_mismatch"


class ProjectionVisibilityRejected(ObservationProjectionError):
    """表示投影视图包含不允许的可见性数据。"""

    code = "projection_visibility_rejected"


class ProjectionSourceChanged(ObservationProjectionError):
    """表示投影构建期间来源记录已变化。"""

    code = "projection_source_changed"


class ProjectionIntegrityFailed(ObservationProjectionError):
    """表示投影完整性校验失败。"""

    code = "projection_integrity_failed"


class ProjectionRenderFailed(ObservationProjectionError):
    """表示投影渲染失败。"""

    code = "projection_render_failed"


class ProjectionBuildFailed(ObservationProjectionError):
    """表示投影构建失败。"""

    code = "projection_build_failed"


__all__ = [
    "ActiveObservationConflict",
    "ObservationProjectionError",
    "ProjectionBuildFailed",
    "ProjectionIdentityMismatch",
    "ProjectionIntegrityFailed",
    "ProjectionRenderFailed",
    "ProjectionSourceChanged",
    "ProjectionVisibilityRejected",
    "RequiredProjectionUnavailable",
]
