# -*- coding: utf-8 -*-
"""
定义由 Host 签发并在提交事务中一次性消费的私密披露授权。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)


class DisclosureGrant(StrictFrozenModel):
    grant_id: NonEmptyId
    actor_id: NonEmptyId
    turn_id: NonEmptyId
    window_id: NonEmptyId
    game_revision: int = Field(ge=0)
    fact_kind: NonEmptyId
    fact_record_id: NonEmptyId
    fact_hash: ContentHash
    target_id: NonEmptyId | None = None
    timing_ref: NonEmptyId
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def _aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        return value
