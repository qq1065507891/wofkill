# -*- coding: utf-8 -*-
"""
验证玩家工作区装配、INDEX 清单与进程内投影缓存。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations

import traceback

import pytest

from werewolf_agent.player_agents.contracts.records import (
    PublicSpeechRecord,
    RecordOrigin,
)
from werewolf_agent.player_agents.contracts.speech import (
    Alignment,
    AlignmentRead,
    Modality,
    Strength,
)
from werewolf_agent.player_agents.observation import (
    CommitmentProjectionSource,
    DocumentRenderer,
    GameProjectionSource,
    InMemoryProjectionCache,
    ObservationAuthoritySnapshot,
    PersonaProjectionSource,
    ProjectedDocument,
    ProjectionAvailability,
    ProjectionCacheKey,
    ProjectionIdentity,
    ProjectionIdentityMismatch,
    ProjectionRenderFailed,
    ProjectionSourceReference,
    PublicSummaryEntry,
    RoleAbilityProjectionSource,
    RoleProjectionSource,
    WorkspaceProjector,
    WorkspaceSection,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _identity(
    *,
    player_id: str = "p01",
    view_fingerprint: str = HASH_A,
) -> ProjectionIdentity:
    return ProjectionIdentity(
        game_id="game-1",
        player_id=player_id,
        schedule_id="schedule-1",
        turn_id="turn-1",
        schedule_state_version=1,
        turn_state_version=2,
        window_id="speech-d1",
        window_version=1,
        base_game_revision=7,
        view_fingerprint=view_fingerprint,
    )


def _source(kind: str, record_id: str, revision: int, content_hash: str) -> ProjectionSourceReference:
    return ProjectionSourceReference(
        record_kind=kind,
        record_id=record_id,
        record_revision=revision,
        content_hash=content_hash,
    )


def _record() -> PublicSpeechRecord:
    return PublicSpeechRecord(
        record_id="speech-1",
        schema_version="1.0.0",
        game_id="game-1",
        turn_id="turn-1",
        actor_id="p01",
        day=1,
        phase="day_discussion",
        committed_revision=7,
        normalized_moves=(
            AlignmentRead(
                move_id="move-1",
                move_type="alignment_read",
                modality=Modality.SUSPECTED,
                evidence_refs=("public-1",),
                target_id="p03",
                alignment=Alignment.WOLF,
                strength=Strength.LEANING,
            ),
        ),
        source_evidence_refs=("public-1",),
        disclosure_grant_refs=(),
        origin=RecordOrigin.MODEL_SUBMISSION,
        renderer_contract_version="speech-renderer-1",
        rendered_utterance_hash=HASH_D,
    )


def _authority_snapshot(
    *,
    commitments_supported: bool = True,
    reordered: bool = False,
) -> ObservationAuthoritySnapshot:
    identity = _identity()
    summaries = (
        PublicSummaryEntry(
            entry_id="summary-b",
            text="p03 的时间线存在矛盾。",
            source_identity=identity,
            source_reference=_source("public_summary", "summary-b", 7, HASH_C),
        ),
        PublicSummaryEntry(
            entry_id="summary-a",
            text="p02 已退出本轮发言。",
            source_identity=identity,
            source_reference=_source("public_summary", "summary-a", 7, HASH_B),
        ),
    )
    commitment = CommitmentProjectionSource(
        record=_record(),
        source_identity=identity,
        source_reference=_source("public_speech", "speech-1", 7, HASH_D),
    )
    return ObservationAuthoritySnapshot(
        identity=identity,
        persona=PersonaProjectionSource(
            profile_id="persona-1",
            profile_version="v3",
            display_name="清醒村民",
            personality_summary="先验证证据，再表达判断。",
            expression_preferences=("简洁", "指出证据编号"),
            risk_appetite="中等",
            verified_tendencies=("愿意修正",),
            source_identity=identity,
            source_reference=_source("persona", "persona-1", 3, HASH_B),
        ),
        role=RoleProjectionSource(
            role_id="villager",
            faction_id="good",
            role_summary="没有夜间技能。",
            abilities=(
                RoleAbilityProjectionSource(
                    ability_id="speak",
                    state="available",
                    restrictions=("仅限白天",),
                ),
            ),
            mechanical_restrictions=("不能查看私密夜间结果",),
            source_identity=identity,
            source_reference=_source("role", "p01-role", 7, HASH_C),
        ),
        game=GameProjectionSource(
            day=1,
            phase="day_discussion",
            living_player_ids=("p03", "p01", "p02"),
            public_summary=tuple(reversed(summaries)) if reordered else summaries,
            authorized_private_fact_references=(),
            source_identity=identity,
            source_references=(_source("game", "game-1", 7, HASH_D),),
        ),
        commitment_records=(commitment,) if commitments_supported else None,
        legal_action_snapshot=("speech",),
        legal_target_snapshot=("p03", "p02"),
        critical_private_fact_references=(),
        bounded_public_summary=("白天讨论继续。",),
        recent_commitment_references=(commitment.source_reference,) if commitments_supported else (),
    )


def _projector(**kwargs: object) -> WorkspaceProjector:
    return WorkspaceProjector(**kwargs)


def _entry(workspace: object, section: WorkspaceSection) -> object:
    return next(
        entry
        for entry in workspace.manifest_entries  # type: ignore[union-attr]
        if entry.section_id is section
    )


def _document(workspace: object, section: WorkspaceSection) -> ProjectedDocument:
    return next(
        document
        for document in workspace.documents  # type: ignore[union-attr]
        if document.section_id is section
    )


def test_workspace_projects_required_and_supported_sections() -> None:
    workspace = _projector().project(_authority_snapshot())
    assert tuple(entry.section_id for entry in workspace.manifest_entries) == tuple(
        WorkspaceSection
    )
    assert _entry(workspace, WorkspaceSection.PLAYER).availability is ProjectionAvailability.AVAILABLE
    assert _entry(workspace, WorkspaceSection.ROLE).availability is ProjectionAvailability.AVAILABLE
    assert _entry(workspace, WorkspaceSection.GAME).availability is ProjectionAvailability.AVAILABLE
    assert _entry(workspace, WorkspaceSection.COMMITMENTS).availability is ProjectionAvailability.AVAILABLE
    assert _entry(workspace, WorkspaceSection.INDEX).availability is ProjectionAvailability.AVAILABLE
    assert _entry(workspace, WorkspaceSection.BELIEFS).unavailable_reason.value == "source_capability_unavailable"
    assert _entry(workspace, WorkspaceSection.MEMORY).source_references == ()
    assert _entry(workspace, WorkspaceSection.WORKING).content_hash is None


def test_workspace_omits_unsupported_commitments_and_index_has_no_self_entry() -> None:
    workspace = _projector().project(_authority_snapshot(commitments_supported=False))
    assert _entry(workspace, WorkspaceSection.COMMITMENTS).availability is ProjectionAvailability.UNAVAILABLE
    index = _document(workspace, WorkspaceSection.INDEX)
    assert "section_id: INDEX.md" not in index.content_markdown
    assert "workspace_revision:" in index.content_markdown


def test_workspace_rebuild_is_byte_identical_after_new_projector_instance() -> None:
    first = WorkspaceProjector().project(_authority_snapshot())
    second = WorkspaceProjector().project(_authority_snapshot(reordered=True))
    assert first.workspace_revision == second.workspace_revision
    assert first.workspace_hash == second.workspace_hash
    assert tuple(document.content_markdown for document in first.documents) == tuple(
        document.content_markdown for document in second.documents
    )


class _FaultingCache:
    def get(self, key: ProjectionCacheKey) -> ProjectedDocument | None:
        raise RuntimeError("private cache read failure")

    def put(self, key: ProjectionCacheKey, document: ProjectedDocument) -> None:
        raise RuntimeError("private cache write failure")


class _CorruptingCache:
    def __init__(self, document: ProjectedDocument) -> None:
        self._document = document.model_copy(update={"content_hash": HASH_E})
        self.replacement: ProjectedDocument | None = None

    def get(self, key: ProjectionCacheKey) -> ProjectedDocument | None:
        return self._document

    def put(self, key: ProjectionCacheKey, document: ProjectedDocument) -> None:
        self.replacement = document


class _WrongIdentityCache:
    def __init__(self, document: ProjectedDocument) -> None:
        self._document = document

    def get(self, key: ProjectionCacheKey) -> ProjectedDocument | None:
        return self._document

    def put(self, key: ProjectionCacheKey, document: ProjectedDocument) -> None:
        raise AssertionError("wrong identity cache value must not be replaced")


def test_cache_failures_become_misses_and_cache_bytes_match_uncached() -> None:
    snapshot = _authority_snapshot()
    uncached = WorkspaceProjector().project(snapshot)
    faulted = WorkspaceProjector(cache=_FaultingCache()).project(snapshot)
    assert tuple(document.content_markdown for document in faulted.documents) == tuple(
        document.content_markdown for document in uncached.documents
    )
    assert faulted.workspace_hash == uncached.workspace_hash


def test_corrupted_cached_document_rebuilds_and_replaces() -> None:
    snapshot = _authority_snapshot()
    original = WorkspaceProjector().project(snapshot)
    cache = _CorruptingCache(_document(original, WorkspaceSection.PLAYER))
    rebuilt = WorkspaceProjector(cache=cache).project(snapshot)
    assert cache.replacement is not None
    assert _document(rebuilt, WorkspaceSection.PLAYER).content_hash != HASH_E


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("player_id", "p02"), ("view_fingerprint", HASH_E)),
)
def test_cross_player_or_wrong_view_cached_document_fails_closed(
    field: str,
    replacement: str,
) -> None:
    snapshot = _authority_snapshot()
    document = _document(WorkspaceProjector().project(snapshot), WorkspaceSection.PLAYER)
    changed_identity = document.identity.model_copy(update={field: replacement})
    cache = _WrongIdentityCache(document.model_copy(update={"identity": changed_identity}))
    with pytest.raises(ProjectionIdentityMismatch):
        WorkspaceProjector(cache=cache).project(snapshot)


def test_in_memory_cache_defensively_copies_caller_documents() -> None:
    snapshot = _authority_snapshot()
    workspace = WorkspaceProjector().project(snapshot)
    document = _document(workspace, WorkspaceSection.PLAYER)
    key = ProjectionCacheKey(
        game_id=snapshot.game_id,
        player_id=snapshot.player_id,
        view_fingerprint=snapshot.view_fingerprint,
        workspace_revision=workspace.workspace_revision,
        section_id=document.section_id,
        renderer_version=document.renderer_version,
        estimator_version=document.estimator_version,
    )
    cache = InMemoryProjectionCache()
    cache.put(key, document)
    object.__setattr__(document, "content_markdown", "# caller mutation\n")
    cached = cache.get(key)
    assert cached is not None
    assert cached.content_markdown != "# caller mutation\n"


class RaisingRenderer:
    section_id = WorkspaceSection.PLAYER
    renderer_version = "raising-v1"

    def render(
        self,
        snapshot: ObservationAuthoritySnapshot,
        estimator: object,
    ) -> ProjectedDocument:
        raise RuntimeError("private-marker")


class ChainedProjectionFailureRenderer:
    """模拟 renderer 把私有内部异常作为稳定错误的原因链暴露。"""

    section_id = WorkspaceSection.PLAYER
    renderer_version = "chained-failure-v1"

    def render(
        self,
        snapshot: ObservationAuthoritySnapshot,
        estimator: object,
    ) -> ProjectedDocument:
        try:
            raise RuntimeError("private-marker")
        except RuntimeError as error:
            raise ProjectionRenderFailed() from error


def test_required_renderer_failure_returns_no_workspace() -> None:
    renderer: DocumentRenderer = RaisingRenderer()
    projector = WorkspaceProjector(renderers={WorkspaceSection.PLAYER: renderer})
    with pytest.raises(ProjectionRenderFailed) as exc_info:
        projector.project(_authority_snapshot())
    assert exc_info.value.__cause__ is None
    assert "private-marker" not in "".join(traceback.format_exception(exc_info.value))


def test_renderer_projection_error_is_replaced_without_private_cause() -> None:
    renderer: DocumentRenderer = ChainedProjectionFailureRenderer()
    projector = WorkspaceProjector(renderers={WorkspaceSection.PLAYER: renderer})
    with pytest.raises(ProjectionRenderFailed) as exc_info:
        projector.project(_authority_snapshot())
    assert exc_info.value.code == "projection_render_failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "private-marker" not in "".join(traceback.format_exception(exc_info.value))
