# -*- coding: utf-8 -*-
"""
定义由 Host 绑定身份与版本的玩家终端提案信封。

作者: Project contributors
创建日期: 2026-07-29
"""

from typing import Literal

from pydantic import Field

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)
from werewolf_agent.player_agents.contracts.speech import SpeechProposalBody


class SpeechProposalEnvelope(StrictFrozenModel):
    schema_version: Literal["1.0.0"]
    turn_id: NonEmptyId
    player_id: NonEmptyId
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    base_revision: int = Field(ge=0)
    view_fingerprint: ContentHash
    body: SpeechProposalBody
