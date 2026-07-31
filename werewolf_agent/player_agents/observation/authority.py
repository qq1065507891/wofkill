# -*- coding: utf-8 -*-
"""
定义观察投影使用的 Host 权威快照及其窄读取边界。

作者: Project contributors
创建日期: 2026-07-31

使用示例:
    >>> isinstance(reader, ObservationAuthorityReader)
    True
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Protocol, Self, runtime_checkable

from pydantic import Field, StringConstraints, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
from werewolf_agent.player_agents.contracts.revisions import ReadReference
from werewolf_agent.player_agents.observation.contracts import (
    BoundedObservationText,
    ProjectionIdentity,
    ProjectionSourceReference,
)

BoundedProjectionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]


def _freeze_list_input(value: object) -> object:
    """把边界列表复制为元组，防止可变输入泄漏进快照。"""

    if isinstance(value, list):
        return tuple(value)
    return value


def _source_key(reference: ProjectionSourceReference) -> tuple[str, str, int]:
    """返回来源身份，不把内容哈希误当作不同来源。"""

    return reference.record_kind, reference.record_id, reference.record_revision


def _validate_unique_source_references(
    references: tuple[ProjectionSourceReference, ...],
    *,
    field_name: str,
) -> tuple[ProjectionSourceReference, ...]:
    """拒绝重复来源或同一来源绑定不同哈希。"""

    hashes: dict[tuple[str, str, int], ContentHash] = {}
    for reference in references:
        key = _source_key(reference)
        previous_hash = hashes.get(key)
        if previous_hash is not None:
            if previous_hash != reference.content_hash:
                raise ValueError("source reference identity has conflicting hashes")
            raise ValueError(f"{field_name} must not contain duplicate sources")
        hashes[key] = reference.content_hash
    return references


def _validate_unique_read_references(
    references: tuple[ReadReference, ...],
    *,
    field_name: str,
) -> tuple[ReadReference, ...]:
    """拒绝重复读取或同一读取身份绑定不同哈希。"""

    hashes: dict[tuple[str, int], ContentHash] = {}
    for reference in references:
        key = reference.record_id, reference.revision
        previous_hash = hashes.get(key)
        if previous_hash is not None:
            if previous_hash != reference.content_hash:
                raise ValueError("read reference identity has conflicting hashes")
            raise ValueError(f"{field_name} must not contain duplicates")
        hashes[key] = reference.content_hash
    return references


class PersonaProjectionSource(StrictFrozenModel):
    """当前查看者已固定的人格资料来源。"""

    profile_id: NonEmptyId
    profile_version: NonEmptyId
    display_name: NonEmptyId
    personality_summary: BoundedProjectionText
    expression_preferences: tuple[BoundedProjectionText, ...] = ()
    risk_appetite: BoundedProjectionText
    verified_tendencies: tuple[BoundedProjectionText, ...] = ()
    source_identity: ProjectionIdentity
    source_reference: ProjectionSourceReference

    @field_validator("expression_preferences", "verified_tendencies", mode="before")
    @classmethod
    def _freeze_collections(cls, value: object) -> object:
        return _freeze_list_input(value)


class RoleAbilityProjectionSource(StrictFrozenModel):
    """当前查看者角色的一项可见能力状态。"""

    ability_id: NonEmptyId
    state: NonEmptyId
    restrictions: tuple[BoundedProjectionText, ...] = ()

    @field_validator("restrictions", mode="before")
    @classmethod
    def _freeze_restrictions(cls, value: object) -> object:
        return _freeze_list_input(value)


class RoleProjectionSource(StrictFrozenModel):
    """当前查看者的自角色事实来源。"""

    role_id: NonEmptyId
    faction_id: NonEmptyId
    role_summary: BoundedProjectionText
    abilities: tuple[RoleAbilityProjectionSource, ...] = ()
    mechanical_restrictions: tuple[BoundedProjectionText, ...] = ()
    source_identity: ProjectionIdentity
    source_reference: ProjectionSourceReference

    @field_validator("abilities", "mechanical_restrictions", mode="before")
    @classmethod
    def _freeze_collections(cls, value: object) -> object:
        return _freeze_list_input(value)

    @model_validator(mode="after")
    def _validate_ability_ids(self) -> Self:
        require_unique(
            (ability.ability_id for ability in self.abilities),
            field_name="role ability IDs",
        )
        return self


class PublicSummaryEntry(StrictFrozenModel):
    """公共摘要的一条 Host 绑定文本。"""

    entry_id: NonEmptyId
    text: BoundedProjectionText
    source_identity: ProjectionIdentity
    source_reference: ProjectionSourceReference


class GameProjectionSource(StrictFrozenModel):
    """当前查看者可见的公共游戏检查点。"""

    day: int = Field(ge=0)
    phase: NonEmptyId
    living_player_ids: tuple[NonEmptyId, ...]
    public_summary: tuple[PublicSummaryEntry, ...] = ()
    authorized_private_fact_references: tuple[ReadReference, ...] = ()
    source_identity: ProjectionIdentity
    source_references: tuple[ProjectionSourceReference, ...] = Field(min_length=1)

    @field_validator(
        "living_player_ids",
        "public_summary",
        "authorized_private_fact_references",
        "source_references",
        mode="before",
    )
    @classmethod
    def _freeze_collections(cls, value: object) -> object:
        return _freeze_list_input(value)

    @model_validator(mode="after")
    def _validate_source_integrity(self) -> Self:
        require_unique(self.living_player_ids, field_name="living_player_ids")
        require_unique(
            (entry.entry_id for entry in self.public_summary),
            field_name="public summary entry IDs",
        )
        _validate_unique_read_references(
            self.authorized_private_fact_references,
            field_name="authorized_private_fact_references",
        )
        _validate_unique_source_references(
            self.source_references,
            field_name="game source_references",
        )
        _validate_unique_source_references(
            tuple(entry.source_reference for entry in self.public_summary),
            field_name="public summary source_references",
        )
        return self


class CommitmentProjectionSource(StrictFrozenModel):
    """已提交公共发言记录及其来源绑定。"""

    record: PublicSpeechRecord
    source_identity: ProjectionIdentity
    source_reference: ProjectionSourceReference

    @model_validator(mode="after")
    def _validate_record_source_match(self) -> Self:
        if (
            self.source_reference.record_id != self.record.record_id
            or self.source_reference.record_revision != self.record.committed_revision
        ):
            raise ValueError("commitment record ID and revision must match source reference")
        return self


class ObservationAuthoritySnapshot(StrictFrozenModel):
    """在单一玩家身份下固定的 Host 过滤观察权威。"""

    identity: ProjectionIdentity
    persona: PersonaProjectionSource
    role: RoleProjectionSource
    game: GameProjectionSource
    commitment_records: tuple[CommitmentProjectionSource, ...] | None
    legal_action_snapshot: tuple[NonEmptyId, ...]
    legal_target_snapshot: tuple[NonEmptyId, ...]
    critical_private_fact_references: tuple[ReadReference, ...]
    bounded_public_summary: tuple[BoundedObservationText, ...]
    recent_commitment_references: tuple[ProjectionSourceReference, ...]
    tool_manifest: tuple[NonEmptyId, ...] = ()

    @field_validator(
        "commitment_records",
        "legal_action_snapshot",
        "legal_target_snapshot",
        "critical_private_fact_references",
        "bounded_public_summary",
        "recent_commitment_references",
        "tool_manifest",
        mode="before",
    )
    @classmethod
    def _freeze_collections(cls, value: object) -> object:
        return _freeze_list_input(value)

    @model_validator(mode="after")
    def _validate_identity_and_source_integrity(self) -> Self:
        if self.tool_manifest:
            raise ValueError("tool_manifest must be empty in this stage")
        require_unique(self.legal_action_snapshot, field_name="legal_action_snapshot")
        require_unique(self.legal_target_snapshot, field_name="legal_target_snapshot")
        _validate_unique_read_references(
            self.critical_private_fact_references,
            field_name="critical_private_fact_references",
        )
        _validate_unique_source_references(
            self.recent_commitment_references,
            field_name="recent_commitment_references",
        )
        source_identities = (
            self.persona.source_identity,
            self.role.source_identity,
            self.game.source_identity,
            *(entry.source_identity for entry in self.game.public_summary),
            *(
                commitment.source_identity
                for commitment in self.commitment_records or ()
            ),
        )
        if any(source_identity != self.identity for source_identity in source_identities):
            raise ValueError("authority source identity must match snapshot identity")

        commitment_references: tuple[ProjectionSourceReference, ...] = ()
        if self.commitment_records is not None:
            for commitment in self.commitment_records:
                if commitment.record.game_id != self.identity.game_id:
                    raise ValueError("commitment record game_id must match identity")
                if commitment.record.actor_id != self.identity.player_id:
                    raise ValueError("commitment records must belong to current viewer")
                if commitment.record.committed_revision > self.identity.base_game_revision:
                    raise ValueError("commitment revision must not exceed base game revision")
            commitment_references = tuple(
                commitment.source_reference for commitment in self.commitment_records
            )
            _validate_unique_source_references(
                commitment_references,
                field_name="commitment record sources",
            )

        committed_sources = set(commitment_references)
        if not all(
            reference in committed_sources
            for reference in self.recent_commitment_references
        ):
            raise ValueError("recent commitment references must be committed sources")

        _validate_unique_read_references(
            (
                *self.game.authorized_private_fact_references,
                *self.critical_private_fact_references,
            ),
            field_name="authority read references",
        )

        _validate_unique_source_references(
            (
                self.persona.source_reference,
                self.role.source_reference,
                *self.game.source_references,
                *(entry.source_reference for entry in self.game.public_summary),
                *commitment_references,
            ),
            field_name="authority source references",
        )
        return self

    @property
    def game_id(self) -> str:
        """暴露已绑定游戏标识，避免调用方拆解 identity。"""

        return self.identity.game_id

    @property
    def player_id(self) -> str:
        """暴露已绑定查看者标识，避免调用方拆解 identity。"""

        return self.identity.player_id

    @property
    def base_game_revision(self) -> int:
        """暴露已绑定基础修订版本。"""

        return self.identity.base_game_revision

    @property
    def view_fingerprint(self) -> str:
        """暴露当前查看者可见性指纹。"""

        return self.identity.view_fingerprint


@runtime_checkable
class ObservationAuthorityReader(Protocol):
    """未来 Host 状态适配器唯一的只读输入缝。"""

    def read_observation_authority(
        self,
        identity: ProjectionIdentity,
        observed_at: datetime,
    ) -> ObservationAuthoritySnapshot: ...


__all__ = [
    "BoundedProjectionText",
    "CommitmentProjectionSource",
    "GameProjectionSource",
    "ObservationAuthorityReader",
    "ObservationAuthoritySnapshot",
    "PersonaProjectionSource",
    "PublicSummaryEntry",
    "RoleAbilityProjectionSource",
    "RoleProjectionSource",
]
