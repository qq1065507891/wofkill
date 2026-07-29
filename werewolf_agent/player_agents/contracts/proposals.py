# -*- coding: utf-8 -*-
"""
定义由 Host 绑定身份与版本的玩家终端提案信封。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-29
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)
from werewolf_agent.player_agents.contracts.speech import (
    ClaimMode,
    RoleClaim,
    SpeechProposalBody,
)


class SpeechProposalEnvelope(StrictFrozenModel):
    schema_version: Literal["1.0.0"]
    turn_id: NonEmptyId
    player_id: NonEmptyId
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    base_revision: int = Field(ge=0)
    view_fingerprint: ContentHash
    body: SpeechProposalBody

    @model_validator(mode="after")
    def _bind_role_claim_actor(self) -> Self:
        for move in self.body.moves:
            if (
                isinstance(move, RoleClaim)
                and move.claim_mode in (ClaimMode.CLAIM, ClaimMode.DENY)
                and move.claimant_id != self.player_id
            ):
                raise ValueError("role claim claimant must match player")
        return self
