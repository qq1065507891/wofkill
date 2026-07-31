# -*- coding: utf-8 -*-
"""
定义自主玩家观察投影、工作区快照和回合观察帧的严格契约。

作者: Project contributors
创建日期: 2026-07-31
修改日期: 2026-07-31

使用示例:
    >>> ProjectionIdentity(
    ...     game_id="game-1", player_id="p01", schedule_id="schedule-1",
    ...     turn_id="turn-1", schedule_state_version=0, turn_state_version=0,
    ...     window_id="speech-d1", window_version=1, base_game_revision=0,
    ...     view_fingerprint="a" * 64,
    ... )
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.revisions import ReadReference


class WorkspaceSection(StrEnum):
    """工作区文档的固定清单顺序。"""

    PLAYER = "PLAYER.md"
    ROLE = "ROLE.md"
    GAME = "GAME.md"
    BELIEFS = "BELIEFS.md"
    COMMITMENTS = "COMMITMENTS.md"
    MEMORY = "MEMORY.md"
    WORKING = "WORKING.md"
    INDEX = "INDEX.md"


_REQUIRED_SECTIONS = frozenset({
    WorkspaceSection.PLAYER,
    WorkspaceSection.ROLE,
    WorkspaceSection.GAME,
    WorkspaceSection.INDEX,
})
_UNAVAILABLE_SECTIONS = frozenset({
    WorkspaceSection.BELIEFS,
    WorkspaceSection.MEMORY,
    WorkspaceSection.WORKING,
})


class ProjectionAvailability(StrEnum):
    """工作区投影是否可供当前观察使用。"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ProjectionVisibilityClass(StrEnum):
    """投影内容在当前玩家视图中的可见性分类。"""

    PLAYER_PRIVATE = "player_private"
    ROLE_PRIVATE = "role_private"
    PUBLIC = "public"
    MIXED_VIEWER_FILTERED = "mixed_viewer_filtered"
    MANIFEST = "manifest"


class ProjectionUnavailableReason(StrEnum):
    """投影不可用的稳定原因。"""

    SOURCE_CAPABILITY_UNAVAILABLE = "source_capability_unavailable"


BoundedObservationText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]


class ProjectionIdentity(StrictFrozenModel):
    """绑定单次玩家视图与调度、回合和游戏版本。"""

    game_id: NonEmptyId
    player_id: NonEmptyId
    schedule_id: NonEmptyId
    turn_id: NonEmptyId
    schedule_state_version: int = Field(ge=0)
    turn_state_version: int = Field(ge=0)
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    base_game_revision: int = Field(ge=0)
    view_fingerprint: ContentHash


class ProjectionSourceReference(StrictFrozenModel):
    """标识用于生成投影的来源记录及其不可变版本。"""

    record_kind: NonEmptyId
    record_id: NonEmptyId
    record_revision: int = Field(ge=0)
    content_hash: ContentHash


def _unique_source_references(
    references: tuple[ProjectionSourceReference, ...],
) -> tuple[ProjectionSourceReference, ...]:
    """拒绝重复或哈希冲突的来源记录身份。"""

    seen: dict[tuple[str, str, int], str] = {}
    for reference in references:
        identity = (
            reference.record_kind,
            reference.record_id,
            reference.record_revision,
        )
        previous_hash = seen.get(identity)
        if previous_hash is not None:
            if previous_hash != reference.content_hash:
                raise ValueError("source reference identity has conflicting hashes")
            raise ValueError("source_references must not contain duplicates")
        seen[identity] = reference.content_hash
    return references


def _require_consistent_source_hashes(
    references: Iterable[ProjectionSourceReference],
) -> None:
    """拒绝同一来源身份在同一工作区引用不同内容。"""

    hashes_by_identity: dict[tuple[str, str, int], str] = {}
    for reference in references:
        identity = (
            reference.record_kind,
            reference.record_id,
            reference.record_revision,
        )
        previous_hash = hashes_by_identity.get(identity)
        if previous_hash is not None and previous_hash != reference.content_hash:
            raise ValueError("workspace source identity has conflicting hashes")
        hashes_by_identity[identity] = reference.content_hash


def _freeze_list_input(value: object) -> object:
    """把边界列表复制为元组，后续仍由严格元素契约校验。"""

    if isinstance(value, list):
        return tuple(value)
    return value


def canonical_json_bytes(value: object) -> bytes:
    """生成内容寻址使用的稳定 UTF-8 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: object) -> str:
    """计算规范 JSON 字节的 SHA-256。"""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_content_hash(content_markdown: str) -> str:
    """计算规范 Markdown UTF-8 字节的 SHA-256。"""

    return hashlib.sha256(content_markdown.encode("utf-8")).hexdigest()


class ProjectedDocument(StrictFrozenModel):
    """单个确定性渲染的工作区文档。"""

    section_id: WorkspaceSection
    identity: ProjectionIdentity
    renderer_version: NonEmptyId
    content_markdown: Annotated[
        str,
        StringConstraints(min_length=1, max_length=65536),
    ]
    content_hash: ContentHash
    token_estimate: int = Field(ge=1)
    estimator_version: NonEmptyId
    visibility_class: ProjectionVisibilityClass
    source_references: tuple[ProjectionSourceReference, ...] = ()

    @field_validator("source_references", mode="before")
    @classmethod
    def _freeze_source_reference_input(cls, value: object) -> object:
        return _freeze_list_input(value)

    @field_validator("content_markdown")
    @classmethod
    def _validate_lf_only_content(cls, content_markdown: str) -> str:
        if "\r" in content_markdown:
            raise ValueError("content_markdown must use LF-only line endings")
        if not content_markdown.endswith("\n") or content_markdown.endswith("\n\n"):
            raise ValueError("content_markdown must have exactly one trailing LF")
        return content_markdown

    @field_validator("source_references")
    @classmethod
    def _validate_source_references(
        cls,
        references: tuple[ProjectionSourceReference, ...],
    ) -> tuple[ProjectionSourceReference, ...]:
        return _unique_source_references(references)

    @model_validator(mode="after")
    def _validate_content_hash(self) -> Self:
        if self.content_hash != canonical_content_hash(self.content_markdown):
            raise ValueError("content_hash must match canonical Markdown bytes")
        return self


class ManifestEntry(StrictFrozenModel):
    """描述一个工作区区段是否可用及其已验证元数据。"""

    section_id: WorkspaceSection
    availability: ProjectionAvailability
    required: bool
    identity: ProjectionIdentity
    renderer_version: NonEmptyId | None
    content_hash: ContentHash | None
    token_estimate: int | None = Field(default=None, ge=1)
    estimator_version: NonEmptyId | None
    visibility_class: ProjectionVisibilityClass
    source_references: tuple[ProjectionSourceReference, ...] = ()
    unavailable_reason: ProjectionUnavailableReason | None = None

    @field_validator("source_references", mode="before")
    @classmethod
    def _freeze_source_reference_input(cls, value: object) -> object:
        return _freeze_list_input(value)

    @field_validator("source_references")
    @classmethod
    def _validate_source_references(
        cls,
        references: tuple[ProjectionSourceReference, ...],
    ) -> tuple[ProjectionSourceReference, ...]:
        return _unique_source_references(references)

    @model_validator(mode="after")
    def _validate_availability(self) -> Self:
        projection_fields_present = all((
            self.renderer_version is not None,
            self.content_hash is not None,
            self.token_estimate is not None,
            self.estimator_version is not None,
        ))
        projection_fields_absent = all((
            self.renderer_version is None,
            self.content_hash is None,
            self.token_estimate is None,
            self.estimator_version is None,
        ))
        if self.availability is ProjectionAvailability.AVAILABLE:
            if not projection_fields_present or self.unavailable_reason is not None:
                raise ValueError("available manifest entry requires projection fields")
        elif (
            self.required
            or not projection_fields_absent
            or self.source_references
            or self.unavailable_reason is None
        ):
            raise ValueError("unavailable manifest entry must omit projection data")
        if self.section_id in _REQUIRED_SECTIONS and (
            self.availability is not ProjectionAvailability.AVAILABLE
            or not self.required
        ):
            raise ValueError("required workspace section must be available")
        if self.section_id in _UNAVAILABLE_SECTIONS and (
            self.availability is not ProjectionAvailability.UNAVAILABLE
            or self.required
        ):
            raise ValueError("workspace section is unavailable in this stage")
        if self.section_id is WorkspaceSection.COMMITMENTS and self.required:
            raise ValueError("COMMITMENTS manifest entry must not be required")
        return self


def canonical_workspace_hash(
    entries: tuple[ManifestEntry, ...],
    documents: tuple[ProjectedDocument, ...],
) -> str:
    """按有序清单和有序文档字节计算工作区 SHA-256。"""

    digest = hashlib.sha256()
    manifest_bytes = canonical_json_bytes([
        entry.model_dump(mode="json") for entry in entries
    ])
    digest.update(len(manifest_bytes).to_bytes(8, "big"))
    digest.update(manifest_bytes)
    for document in documents:
        content_bytes = document.content_markdown.encode("utf-8")
        digest.update(len(content_bytes).to_bytes(8, "big"))
        digest.update(content_bytes)
    return digest.hexdigest()


class PlayerWorkspaceSnapshot(StrictFrozenModel):
    """在一个投影身份下的完整玩家工作区快照。"""

    identity: ProjectionIdentity
    workspace_revision: ContentHash
    documents: tuple[ProjectedDocument, ...]
    manifest_entries: tuple[ManifestEntry, ...]
    workspace_hash: ContentHash

    @field_validator("documents", "manifest_entries", mode="before")
    @classmethod
    def _freeze_collection_input(cls, value: object) -> object:
        return _freeze_list_input(value)

    @model_validator(mode="after")
    def _validate_workspace(self) -> Self:
        expected_sections = tuple(WorkspaceSection)
        actual_sections = tuple(entry.section_id for entry in self.manifest_entries)
        if actual_sections != expected_sections:
            raise ValueError("manifest entries must cover sections in enum order")

        entries_by_section = {
            entry.section_id: entry for entry in self.manifest_entries
        }
        if any(entry.identity != self.identity for entry in self.manifest_entries):
            raise ValueError("manifest identities must match workspace identity")
        documents_by_section = {document.section_id: document for document in self.documents}
        if len(documents_by_section) != len(self.documents):
            raise ValueError("documents must not contain duplicate sections")

        available_sections = {
            entry.section_id
            for entry in self.manifest_entries
            if entry.availability is ProjectionAvailability.AVAILABLE
        }
        if set(documents_by_section) != available_sections:
            raise ValueError("documents must match available manifest entries")
        expected_document_sections = tuple(
            entry.section_id
            for entry in self.manifest_entries
            if entry.availability is ProjectionAvailability.AVAILABLE
        )
        actual_document_sections = tuple(
            document.section_id for document in self.documents
        )
        if actual_document_sections != expected_document_sections:
            raise ValueError("documents must follow available manifest order")

        for section_id, document in documents_by_section.items():
            entry = entries_by_section[section_id]
            if (
                document.identity != self.identity
                or document.identity != entry.identity
                or document.renderer_version != entry.renderer_version
                or document.content_hash != entry.content_hash
                or document.token_estimate != entry.token_estimate
                or document.estimator_version != entry.estimator_version
                or document.visibility_class != entry.visibility_class
                or document.source_references != entry.source_references
            ):
                raise ValueError("document metadata must match its manifest entry")
        source_references: list[ProjectionSourceReference] = []
        for document in self.documents:
            source_references.extend(document.source_references)
        for entry in self.manifest_entries:
            source_references.extend(entry.source_references)
        _require_consistent_source_hashes(source_references)
        if self.workspace_hash != canonical_workspace_hash(
            self.manifest_entries,
            self.documents,
        ):
            raise ValueError("workspace_hash must match ordered workspace bytes")
        return self


class ObservationFrame(StrictFrozenModel):
    """模型调用前固定的玩家观察、权限和时间边界。"""

    identity: ProjectionIdentity
    task_kind: Literal["day_speech"]
    actor_id: NonEmptyId
    role_id: NonEmptyId
    phase: Literal["day_discussion"]
    legal_action_snapshot: tuple[NonEmptyId, ...]
    legal_target_snapshot: tuple[NonEmptyId, ...]
    critical_private_fact_references: tuple[ReadReference, ...]
    bounded_public_summary: tuple[BoundedObservationText, ...]
    recent_commitment_references: tuple[ProjectionSourceReference, ...]
    document_manifest: tuple[ManifestEntry, ...]
    tool_manifest: tuple[NonEmptyId, ...] = ()
    workspace_revision: ContentHash
    workspace_hash: ContentHash
    deadline: datetime
    observed_at: datetime

    @field_validator(
        "legal_action_snapshot",
        "legal_target_snapshot",
        "critical_private_fact_references",
        "bounded_public_summary",
        "recent_commitment_references",
        "document_manifest",
        "tool_manifest",
        mode="before",
    )
    @classmethod
    def _freeze_collection_input(cls, value: object) -> object:
        return _freeze_list_input(value)

    @field_validator(
        "legal_action_snapshot",
        "legal_target_snapshot",
        "critical_private_fact_references",
        "tool_manifest",
    )
    @classmethod
    def _validate_unique_tuples(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        return require_unique(values, field_name="observation frame tuple")

    @field_validator("recent_commitment_references")
    @classmethod
    def _validate_recent_commitment_references(
        cls,
        references: tuple[ProjectionSourceReference, ...],
    ) -> tuple[ProjectionSourceReference, ...]:
        return _unique_source_references(references)

    @model_validator(mode="after")
    def _validate_bound_context(self) -> Self:
        if self.actor_id != self.identity.player_id:
            raise ValueError("actor_id must match projection identity player_id")
        if not _is_aware(self.deadline) or not _is_aware(self.observed_at):
            raise ValueError("deadline and observed_at must be timezone-aware")
        if self.observed_at >= self.deadline:
            raise ValueError("observed_at must be before deadline")
        return self


class ObservationBundle(StrictFrozenModel):
    """绑定回合观察帧与完整工作区快照的不可变输入。"""

    frame: ObservationFrame
    workspace: PlayerWorkspaceSnapshot

    @model_validator(mode="after")
    def _validate_frame_workspace_match(self) -> Self:
        if (
            self.frame.identity != self.workspace.identity
            or self.frame.workspace_revision != self.workspace.workspace_revision
            or self.frame.workspace_hash != self.workspace.workspace_hash
            or self.frame.document_manifest != self.workspace.manifest_entries
        ):
            raise ValueError("frame must match its workspace snapshot")
        return self


def _is_aware(value: datetime) -> bool:
    """判断时间是否携带可比较的时区偏移。"""

    return value.tzinfo is not None and value.utcoffset() is not None


__all__ = [
    "BoundedObservationText",
    "ManifestEntry",
    "ObservationBundle",
    "ObservationFrame",
    "PlayerWorkspaceSnapshot",
    "ProjectedDocument",
    "ProjectionAvailability",
    "ProjectionIdentity",
    "ProjectionSourceReference",
    "ProjectionUnavailableReason",
    "ProjectionVisibilityClass",
    "WorkspaceSection",
    "canonical_content_hash",
    "canonical_json_bytes",
    "canonical_json_hash",
    "canonical_workspace_hash",
]
