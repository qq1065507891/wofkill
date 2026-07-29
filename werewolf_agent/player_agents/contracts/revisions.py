# -*- coding: utf-8 -*-
"""
定义终端提案与工具读取共用的修订版本和读取集合契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from pydantic import Field

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)


class ReadReference(StrictFrozenModel):
    """把一次读取绑定到不可变记录、产生版本和内容哈希。"""

    record_id: NonEmptyId
    revision: int = Field(ge=0)
    content_hash: ContentHash


class RevisionContext(StrictFrozenModel):
    """绑定提案所依据的游戏、窗口和查看者视图版本。"""

    base_revision: int = Field(ge=0)
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    view_fingerprint: ContentHash
