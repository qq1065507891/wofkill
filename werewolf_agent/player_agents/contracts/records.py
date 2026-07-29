# -*- coding: utf-8 -*-
"""
定义提交后的公共发言语义记录和无独立语义权威的渲染结果。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-29
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.speech import (
    PrivateResultDisclosure,
    SpeechMove,
)


class RecordOrigin(StrEnum):
    MODEL_SUBMISSION = "model_submission"
    REPAIRED_SUBMISSION = "repaired_submission"
    NEUTRAL_TERMINAL_FALLBACK = "neutral_terminal_fallback"


class PublicSpeechRecord(StrictFrozenModel):
    record_id: NonEmptyId
    schema_version: Literal["1.0.0"]
    game_id: NonEmptyId
    turn_id: NonEmptyId
    actor_id: NonEmptyId
    day: int = Field(ge=0)
    phase: NonEmptyId
    committed_revision: int = Field(ge=1)
    normalized_moves: tuple[SpeechMove, ...] = Field(min_length=1, max_length=8)
    source_evidence_refs: tuple[NonEmptyId, ...] = ()
    disclosure_grant_refs: tuple[NonEmptyId, ...] = ()
    origin: RecordOrigin
    renderer_contract_version: NonEmptyId
    rendered_utterance_hash: ContentHash

    @model_validator(mode="after")
    def _consistent_provenance(self) -> "PublicSpeechRecord":
        require_unique(
            (move.move_id for move in self.normalized_moves),
            field_name="move IDs",
        )
        require_unique(self.source_evidence_refs, field_name="source_evidence_refs")
        require_unique(self.disclosure_grant_refs, field_name="disclosure_grant_refs")
        move_evidence_refs = {
            evidence_ref
            for move in self.normalized_moves
            for evidence_ref in move.evidence_refs
        }
        if not move_evidence_refs <= set(self.source_evidence_refs):
            raise ValueError("source_evidence_refs must include every move evidence ref")
        used_grant_refs = {
            move.disclosure_grant_id
            for move in self.normalized_moves
            if isinstance(move, PrivateResultDisclosure)
        }
        if set(self.disclosure_grant_refs) != used_grant_refs:
            raise ValueError(
                "disclosure_grant_refs must match private disclosure moves"
            )
        return self


class RenderedUtterance(StrictFrozenModel):
    record_id: NonEmptyId
    sentence_plan_version: NonEmptyId
    renderer_version: NonEmptyId
    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    content_hash: ContentHash
    fallback_status: Literal["none", "template_fallback"]

    @field_validator("text")
    @classmethod
    def _non_whitespace_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain a non-whitespace character")
        return value
