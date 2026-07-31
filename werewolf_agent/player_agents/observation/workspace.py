# -*- coding: utf-8 -*-
"""
装配有序玩家工作区、INDEX 清单和可选的派生文档缓存。

作者: Project contributors
创建日期: 2026-07-31

使用示例:
    >>> WorkspaceProjector().project(snapshot)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from werewolf_agent.player_agents.observation.authority import (
    ObservationAuthoritySnapshot,
)
from werewolf_agent.player_agents.observation.cache import (
    ProjectionCache,
    ProjectionCacheKey,
)
from werewolf_agent.player_agents.observation.contracts import (
    ManifestEntry,
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
    ObservationProjectionError,
    ProjectionIdentityMismatch,
    ProjectionIntegrityFailed,
    ProjectionRenderFailed,
    RequiredProjectionUnavailable,
)
from werewolf_agent.player_agents.observation.rendering import (
    DOCUMENT_RENDERERS,
    ConservativeTokenEstimator,
    DocumentRenderer,
    TokenEstimator,
)

INDEX_RENDERER_VERSION = "index-v1"

_REQUIRED_SECTIONS = frozenset({
    WorkspaceSection.PLAYER,
    WorkspaceSection.ROLE,
    WorkspaceSection.GAME,
    WorkspaceSection.INDEX,
})
_ALWAYS_UNAVAILABLE_SECTIONS = frozenset({
    WorkspaceSection.BELIEFS,
    WorkspaceSection.MEMORY,
    WorkspaceSection.WORKING,
})


@dataclass(frozen=True)
class _SectionPlan:
    """投影前固定一个区段的可用性、来源和渲染版本。"""

    section_id: WorkspaceSection
    availability: ProjectionAvailability
    required: bool
    renderer_version: str | None
    visibility_class: ProjectionVisibilityClass
    source_references: tuple[ProjectionSourceReference, ...]


def _canonical_json_bytes(value: object) -> bytes:
    """生成用于内容寻址的稳定 UTF-8 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sources(
    references: tuple[ProjectionSourceReference, ...],
) -> tuple[ProjectionSourceReference, ...]:
    """按完整来源身份排序，隔离上游收集顺序。"""

    return tuple(sorted(
        references,
        key=lambda item: (
            item.record_kind,
            item.record_id,
            item.record_revision,
            item.content_hash,
        ),
    ))


def _source_payload(
    references: tuple[ProjectionSourceReference, ...],
) -> list[dict[str, Any]]:
    """将来源引用转换为 workspace revision 的规范输入。"""

    return [
        reference.model_dump(mode="json")
        for reference in _canonical_sources(references)
    ]


def _content_hash(content: str) -> str:
    """计算规范 Markdown 的 UTF-8 内容哈希。"""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _safe_scalar(value: str) -> str:
    """将标量写成单行 JSON 字面量，不能改变 INDEX 结构。"""

    return json.dumps(value, ensure_ascii=False)


class WorkspaceProjector:
    """把单一权威快照投影为完整、不可变的逻辑工作区。"""

    def __init__(
        self,
        *,
        renderers: Mapping[WorkspaceSection, DocumentRenderer] | None = None,
        estimator: TokenEstimator | None = None,
        cache: ProjectionCache | None = None,
    ) -> None:
        configured_renderers = dict(DOCUMENT_RENDERERS)
        if renderers is not None:
            configured_renderers.update(renderers)
        self._renderers = configured_renderers
        self._estimator = estimator or ConservativeTokenEstimator()
        self._cache = cache

    def project(
        self,
        snapshot: ObservationAuthoritySnapshot,
    ) -> PlayerWorkspaceSnapshot:
        """验证权威输入后，构建完整的确定性逻辑工作区。"""

        snapshot = self._validate_snapshot(snapshot)
        plans = self._plans(snapshot)
        workspace_revision = self._workspace_revision(snapshot.identity, plans)
        documents: list[ProjectedDocument] = []
        entries: list[ManifestEntry] = []

        for plan in plans:
            if plan.section_id is WorkspaceSection.INDEX:
                continue
            if plan.availability is ProjectionAvailability.UNAVAILABLE:
                entries.append(self._unavailable_entry(snapshot.identity, plan))
                continue
            document = self._cached_or_rendered(
                snapshot,
                plan,
                workspace_revision,
            )
            documents.append(document)
            entries.append(self._available_entry(document, required=plan.required))

        index_document = self._render_index(
            identity=snapshot.identity,
            workspace_revision=workspace_revision,
            pre_index_entries=tuple(entries),
        )
        documents.append(index_document)
        entries.append(self._available_entry(index_document, required=True))
        manifest_entries = tuple(entries)
        workspace_hash = self._workspace_hash(manifest_entries, tuple(documents))
        try:
            return PlayerWorkspaceSnapshot(
                identity=snapshot.identity,
                workspace_revision=workspace_revision,
                documents=tuple(documents),
                manifest_entries=manifest_entries,
                workspace_hash=workspace_hash,
            )
        except ValidationError:
            raise ProjectionIntegrityFailed() from None

    def _validate_snapshot(
        self,
        snapshot: ObservationAuthoritySnapshot,
    ) -> ObservationAuthoritySnapshot:
        """重新验证可能经由 model_copy 传播的权威边界。"""

        try:
            return ObservationAuthoritySnapshot.model_validate(
                snapshot.model_dump(round_trip=True)
            )
        except (AttributeError, TypeError, ValidationError):
            raise ProjectionIntegrityFailed() from None

    def _plans(
        self,
        snapshot: ObservationAuthoritySnapshot,
    ) -> tuple[_SectionPlan, ...]:
        """冻结所有区段可用性及其唯一允许的来源集合。"""

        plans: list[_SectionPlan] = []
        for section_id in WorkspaceSection:
            if section_id in _ALWAYS_UNAVAILABLE_SECTIONS:
                plans.append(_SectionPlan(
                    section_id=section_id,
                    availability=ProjectionAvailability.UNAVAILABLE,
                    required=False,
                    renderer_version=None,
                    visibility_class=ProjectionVisibilityClass.MANIFEST,
                    source_references=(),
                ))
                continue
            if section_id is WorkspaceSection.INDEX:
                plans.append(_SectionPlan(
                    section_id=section_id,
                    availability=ProjectionAvailability.AVAILABLE,
                    required=True,
                    renderer_version=INDEX_RENDERER_VERSION,
                    visibility_class=ProjectionVisibilityClass.MANIFEST,
                    source_references=(),
                ))
                continue
            if (
                section_id is WorkspaceSection.COMMITMENTS
                and snapshot.commitment_records is None
            ):
                plans.append(_SectionPlan(
                    section_id=section_id,
                    availability=ProjectionAvailability.UNAVAILABLE,
                    required=False,
                    renderer_version=None,
                    visibility_class=ProjectionVisibilityClass.MANIFEST,
                    source_references=(),
                ))
                continue
            renderer = self._renderers.get(section_id)
            if renderer is None:
                if section_id in _REQUIRED_SECTIONS:
                    raise RequiredProjectionUnavailable()
                plans.append(_SectionPlan(
                    section_id=section_id,
                    availability=ProjectionAvailability.UNAVAILABLE,
                    required=False,
                    renderer_version=None,
                    visibility_class=ProjectionVisibilityClass.MANIFEST,
                    source_references=(),
                ))
                continue
            plans.append(_SectionPlan(
                section_id=section_id,
                availability=ProjectionAvailability.AVAILABLE,
                required=section_id in _REQUIRED_SECTIONS,
                renderer_version=renderer.renderer_version,
                visibility_class=self._visibility_for(section_id),
                source_references=self._sources_for(snapshot, section_id),
            ))
        return tuple(plans)

    def _sources_for(
        self,
        snapshot: ObservationAuthoritySnapshot,
        section_id: WorkspaceSection,
    ) -> tuple[ProjectionSourceReference, ...]:
        """返回指定 renderer 明确允许读取的实际来源。"""

        if section_id is WorkspaceSection.PLAYER:
            return (snapshot.persona.source_reference,)
        if section_id is WorkspaceSection.ROLE:
            return (snapshot.role.source_reference,)
        if section_id is WorkspaceSection.GAME:
            return (
                *snapshot.game.source_references,
                *(entry.source_reference for entry in snapshot.game.public_summary),
            )
        if section_id is WorkspaceSection.COMMITMENTS:
            return tuple(
                item.source_reference
                for item in snapshot.commitment_records or ()
            )
        raise ProjectionIntegrityFailed()

    @staticmethod
    def _visibility_for(section_id: WorkspaceSection) -> ProjectionVisibilityClass:
        """保持 renderer 的固定可见性契约与 workspace 元数据一致。"""

        visibility = {
            WorkspaceSection.PLAYER: ProjectionVisibilityClass.PLAYER_PRIVATE,
            WorkspaceSection.ROLE: ProjectionVisibilityClass.ROLE_PRIVATE,
            WorkspaceSection.GAME: ProjectionVisibilityClass.PUBLIC,
            WorkspaceSection.COMMITMENTS: ProjectionVisibilityClass.MIXED_VIEWER_FILTERED,
        }
        return visibility[section_id]

    def _workspace_revision(
        self,
        identity: ProjectionIdentity,
        plans: tuple[_SectionPlan, ...],
    ) -> str:
        """在任何文档查询前固定 workspace 的内容寻址修订版本。"""

        payload = {
            "projection_identity": identity.model_dump(mode="json"),
            "sections": [
                {
                    "section_id": plan.section_id,
                    "availability": plan.availability,
                    "source_references": _source_payload(plan.source_references),
                    "renderer_version": plan.renderer_version,
                }
                for plan in plans
            ],
            "estimator_version": self._estimator.version,
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def _cached_or_rendered(
        self,
        snapshot: ObservationAuthoritySnapshot,
        plan: _SectionPlan,
        workspace_revision: str,
    ) -> ProjectedDocument:
        """只接受完整匹配的缓存值，否则重建并尽力回填。"""

        assert plan.renderer_version is not None
        key = ProjectionCacheKey(
            game_id=snapshot.game_id,
            player_id=snapshot.player_id,
            view_fingerprint=snapshot.view_fingerprint,
            workspace_revision=workspace_revision,
            section_id=plan.section_id,
            renderer_version=plan.renderer_version,
            estimator_version=self._estimator.version,
        )
        cached = self._cache_get(key)
        if cached is not None:
            self._validate_cached_document(cached, snapshot.identity, plan, key)
            if self._is_complete_document(cached, snapshot.identity, plan):
                return cached
        document = self._render(snapshot, plan)
        self._cache_put(key, document)
        return document

    def _cache_get(self, key: ProjectionCacheKey) -> ProjectedDocument | None:
        """把意外 cache 故障降级为 miss，保持缓存可选。"""

        if self._cache is None:
            return None
        try:
            return self._cache.get(key)
        except ProjectionIdentityMismatch:
            raise
        except Exception:  # noqa: BLE001 - 可选缓存的任何故障都必须降级为 miss。
            return None

    def _cache_put(self, key: ProjectionCacheKey, document: ProjectedDocument) -> None:
        """缓存写入失败不影响当前已验证的渲染结果。"""

        if self._cache is None:
            return
        try:
            self._cache.put(key, document)
        except ProjectionIdentityMismatch:
            raise
        except Exception:  # noqa: BLE001 - 写缓存不能影响已验证的当前结果。
            return

    def _validate_cached_document(
        self,
        document: ProjectedDocument,
        identity: ProjectionIdentity,
        plan: _SectionPlan,
        key: ProjectionCacheKey,
    ) -> None:
        """身份越界立即失败，其余缓存破损交由重建路径处理。"""

        if not isinstance(document, ProjectedDocument):
            return
        document_identity = document.identity
        if (
            document_identity.game_id != identity.game_id
            or document_identity.player_id != identity.player_id
            or document_identity.view_fingerprint != identity.view_fingerprint
        ):
            raise ProjectionIdentityMismatch()

    def _is_complete_document(
        self,
        document: ProjectedDocument,
        identity: ProjectionIdentity,
        plan: _SectionPlan,
    ) -> bool:
        """检查缓存文档能否作为本次唯一权威投影的派生结果。"""

        if not isinstance(document, ProjectedDocument):
            return False
        return (
            document.identity == identity
            and document.section_id is plan.section_id
            and document.renderer_version == plan.renderer_version
            and document.estimator_version == self._estimator.version
            and document.visibility_class is plan.visibility_class
            and document.source_references == _canonical_sources(plan.source_references)
            and document.content_hash == _content_hash(document.content_markdown)
            and document.token_estimate == self._estimator.estimate(document.content_markdown)
        )

    def _render(
        self,
        snapshot: ObservationAuthoritySnapshot,
        plan: _SectionPlan,
    ) -> ProjectedDocument:
        """调用指定 renderer，并将实现异常净化为稳定错误。"""

        renderer = self._renderers.get(plan.section_id)
        if renderer is None:
            raise RequiredProjectionUnavailable()
        try:
            document = renderer.render(snapshot, self._estimator)
        except ObservationProjectionError:
            raise
        except Exception:  # noqa: BLE001 - renderer 实现细节不得泄漏给调用方。
            raise ProjectionRenderFailed() from None
        if not self._is_complete_document(document, snapshot.identity, plan):
            raise ProjectionIntegrityFailed()
        return document

    @staticmethod
    def _available_entry(
        document: ProjectedDocument,
        *,
        required: bool,
    ) -> ManifestEntry:
        """从已验证文档生成完全匹配的可用清单条目。"""

        return ManifestEntry(
            section_id=document.section_id,
            availability=ProjectionAvailability.AVAILABLE,
            required=required,
            identity=document.identity,
            renderer_version=document.renderer_version,
            content_hash=document.content_hash,
            token_estimate=document.token_estimate,
            estimator_version=document.estimator_version,
            visibility_class=document.visibility_class,
            source_references=document.source_references,
        )

    @staticmethod
    def _unavailable_entry(
        identity: ProjectionIdentity,
        plan: _SectionPlan,
    ) -> ManifestEntry:
        """为当前阶段尚无权威能力的区段发射通用占位条目。"""

        return ManifestEntry(
            section_id=plan.section_id,
            availability=ProjectionAvailability.UNAVAILABLE,
            required=False,
            identity=identity,
            renderer_version=None,
            content_hash=None,
            token_estimate=None,
            estimator_version=None,
            visibility_class=ProjectionVisibilityClass.MANIFEST,
            source_references=(),
            unavailable_reason=ProjectionUnavailableReason.SOURCE_CAPABILITY_UNAVAILABLE,
        )

    def _render_index(
        self,
        *,
        identity: ProjectionIdentity,
        workspace_revision: str,
        pre_index_entries: tuple[ManifestEntry, ...],
    ) -> ProjectedDocument:
        """从 pre-INDEX manifest 渲染 INDEX，绝不读 INDEX 自身条目。"""

        lines = [
            "# INDEX.md",
            f"- workspace_revision: {workspace_revision}",
            "## SECTIONS",
        ]
        for entry in pre_index_entries:
            lines.extend((
                f"- section_id: {_safe_scalar(entry.section_id)}",
                f"  - availability: {_safe_scalar(entry.availability)}",
                f"  - required: {str(entry.required).lower()}",
                f"  - renderer_version: {_safe_scalar(entry.renderer_version or '-')}",
                f"  - content_hash: {_safe_scalar(entry.content_hash or '-')}",
                f"  - token_estimate: {entry.token_estimate if entry.token_estimate is not None else '-'}",
                f"  - estimator_version: {_safe_scalar(entry.estimator_version or '-')}",
                f"  - visibility_class: {_safe_scalar(entry.visibility_class)}",
                "  - source_ids:",
                *(
                    f"    - {_safe_scalar(f'{source.record_kind}/{source.record_id}@{source.record_revision}') }"
                    for source in entry.source_references
                ),
                f"  - unavailable_reason: {_safe_scalar(entry.unavailable_reason or '-')}",
            ))
        content = "\n".join(lines) + "\n"
        return ProjectedDocument(
            section_id=WorkspaceSection.INDEX,
            identity=identity,
            renderer_version=INDEX_RENDERER_VERSION,
            content_markdown=content,
            content_hash=_content_hash(content),
            token_estimate=self._estimator.estimate(content),
            estimator_version=self._estimator.version,
            visibility_class=ProjectionVisibilityClass.MANIFEST,
            source_references=(),
        )

    @staticmethod
    def _workspace_hash(
        entries: tuple[ManifestEntry, ...],
        documents: tuple[ProjectedDocument, ...],
    ) -> str:
        """按清单顺序和文档 UTF-8 字节计算不可歧义的 workspace 哈希。"""

        digest = hashlib.sha256()
        manifest_bytes = _canonical_json_bytes([
            entry.model_dump(mode="json") for entry in entries
        ])
        digest.update(len(manifest_bytes).to_bytes(8, "big"))
        digest.update(manifest_bytes)
        for document in documents:
            content_bytes = document.content_markdown.encode("utf-8")
            digest.update(len(content_bytes).to_bytes(8, "big"))
            digest.update(content_bytes)
        return digest.hexdigest()


__all__ = ["INDEX_RENDERER_VERSION", "WorkspaceProjector"]
