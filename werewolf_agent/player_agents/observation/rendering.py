# -*- coding: utf-8 -*-
"""
将 Host 过滤权威快照渲染为确定性的玩家工作区文档。

作者: Project contributors
创建日期: 2026-07-31

使用示例:
    >>> ConservativeTokenEstimator().estimate("# PLAYER.md\n") >= 1
    True
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from werewolf_agent.player_agents.contracts.speech import (
    AlignmentRead,
    ConditionalCommitment,
    PlayerComparison,
    PrivateResultDisclosure,
    PublicEvidenceCitation,
    QuestionMove,
    ResponseMove,
    RetractionMove,
    RoleClaim,
    SpeechMove,
    UncertaintyStatement,
    VotePosition,
)
from werewolf_agent.player_agents.observation.authority import (
    CommitmentProjectionSource,
    ObservationAuthoritySnapshot,
)
from werewolf_agent.player_agents.observation.contracts import (
    ProjectedDocument,
    ProjectionIdentity,
    ProjectionSourceReference,
    ProjectionVisibilityClass,
    WorkspaceSection,
)

PLAYER_RENDERER_VERSION = "player-v1"
ROLE_RENDERER_VERSION = "role-v1"
GAME_RENDERER_VERSION = "game-v1"
COMMITMENTS_RENDERER_VERSION = "commitments-v1"


@runtime_checkable
class TokenEstimator(Protocol):
    """为规范文档提供稳定 token 估算的窄协议。"""

    version: str

    def estimate(self, text: str) -> int: ...


@runtime_checkable
class DocumentRenderer(Protocol):
    """将单一权威快照投影为一个固定工作区区段。"""

    section_id: WorkspaceSection
    renderer_version: str

    def render(
        self,
        snapshot: ObservationAuthoritySnapshot,
        estimator: TokenEstimator,
    ) -> ProjectedDocument: ...


class ConservativeTokenEstimator:
    """按 Unicode 与 UTF-8 字节数保守估算 token。"""

    version = "unicode-conservative-v1"

    def estimate(self, text: str) -> int:
        codepoints = len(text)
        byte_quarters = math.ceil(len(text.encode("utf-8")) / 4)
        baseline = max(codepoints, byte_quarters, 1)
        return math.ceil(baseline * 1.10)


def _canonical_source_order(
    references: tuple[ProjectionSourceReference, ...],
) -> tuple[ProjectionSourceReference, ...]:
    """以来源身份排序，确保输入收集顺序不影响文档字节。"""

    return tuple(sorted(
        references,
        key=lambda item: (
            item.record_kind,
            item.record_id,
            item.record_revision,
            item.content_hash,
        ),
    ))


def _canonical_document(title: str, lines: list[str]) -> str:
    """生成只含 LF 且恰有一个末尾换行的固定 Markdown。"""

    return "\n".join((f"# {title}", *lines)).rstrip("\n") + "\n"


def _escape_untrusted_text(text: str) -> str:
    """把公共自由文本限制为引用数据，不能产生 Markdown 控制结构。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    without_controls = "".join(
        character
        for character in normalized
        if character == "\n" or ord(character) >= 32 and ord(character) != 127
    )
    return (
        without_controls.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _quoted_untrusted_lines(text: str) -> list[str]:
    """逐行引用已转义的公共文本，包括空行。"""

    return [f"> {line}" for line in _escape_untrusted_text(text).split("\n")]


def _source_lines(references: tuple[ProjectionSourceReference, ...]) -> list[str]:
    """稳定显示当前文档实际依赖的来源身份与哈希。"""

    return [
        (
            "- source: "
            f"{reference.record_kind}/{reference.record_id}"
            f"@{reference.record_revision} sha256:{reference.content_hash}"
        )
        for reference in _canonical_source_order(references)
    ]


def _build_document(
    *,
    section_id: WorkspaceSection,
    renderer_version: str,
    identity: ProjectionIdentity,
    content_markdown: str,
    estimator: TokenEstimator,
    visibility_class: ProjectionVisibilityClass,
    source_references: tuple[ProjectionSourceReference, ...],
) -> ProjectedDocument:
    """以统一哈希和元数据构建后再次核验投影文档。"""

    canonical_references = _canonical_source_order(source_references)
    document = ProjectedDocument(
        section_id=section_id,
        identity=identity,
        renderer_version=renderer_version,
        content_markdown=content_markdown,
        content_hash=hashlib.sha256(content_markdown.encode("utf-8")).hexdigest(),
        token_estimate=estimator.estimate(content_markdown),
        estimator_version=estimator.version,
        visibility_class=visibility_class,
        source_references=canonical_references,
    )
    if (
        document.identity != identity
        or document.source_references != canonical_references
        or document.content_hash
        != hashlib.sha256(document.content_markdown.encode("utf-8")).hexdigest()
    ):
        raise ValueError("renderer produced an invalid projected document")
    return document


def render_player_document(
    snapshot: ObservationAuthoritySnapshot,
    estimator: TokenEstimator,
) -> ProjectedDocument:
    """渲染人格资料；只访问身份和人格来源，不读取角色或游戏字段。"""

    persona = snapshot.persona
    content = _canonical_document("PLAYER.md", [
        "## PROFILE",
        f"- profile_id: {persona.profile_id}",
        f"- profile_version: {persona.profile_version}",
        f"- display_name: {persona.display_name}",
        f"- personality_summary: {persona.personality_summary}",
        f"- risk_appetite: {persona.risk_appetite}",
        "## EXPRESSION_PREFERENCES",
        *(f"- {value}" for value in sorted(persona.expression_preferences)),
        "## VERIFIED_TENDENCIES",
        *(f"- {value}" for value in sorted(persona.verified_tendencies)),
        "## SOURCES",
        *_source_lines((persona.source_reference,)),
    ])
    return _build_document(
        section_id=WorkspaceSection.PLAYER,
        renderer_version=PLAYER_RENDERER_VERSION,
        identity=snapshot.identity,
        content_markdown=content,
        estimator=estimator,
        visibility_class=ProjectionVisibilityClass.PLAYER_PRIVATE,
        source_references=(persona.source_reference,),
    )


def render_role_document(
    snapshot: ObservationAuthoritySnapshot,
    estimator: TokenEstimator,
) -> ProjectedDocument:
    """渲染当前查看者的自角色与可见机制限制。"""

    role = snapshot.role
    ability_lines: list[str] = []
    for ability in sorted(role.abilities, key=lambda item: item.ability_id):
        ability_lines.extend((
            f"- ability_id: {ability.ability_id}",
            f"  - state: {ability.state}",
            *(f"  - restriction: {value}" for value in sorted(ability.restrictions)),
        ))
    content = _canonical_document("ROLE.md", [
        "## SELF_ROLE",
        f"- role_id: {role.role_id}",
        f"- faction_id: {role.faction_id}",
        f"- role_summary: {role.role_summary}",
        "## ABILITIES",
        *ability_lines,
        "## MECHANICAL_RESTRICTIONS",
        *(f"- {value}" for value in sorted(role.mechanical_restrictions)),
        "## SOURCES",
        *_source_lines((role.source_reference,)),
    ])
    return _build_document(
        section_id=WorkspaceSection.ROLE,
        renderer_version=ROLE_RENDERER_VERSION,
        identity=snapshot.identity,
        content_markdown=content,
        estimator=estimator,
        visibility_class=ProjectionVisibilityClass.ROLE_PRIVATE,
        source_references=(role.source_reference,),
    )


def render_game_document(
    snapshot: ObservationAuthoritySnapshot,
    estimator: TokenEstimator,
) -> ProjectedDocument:
    """渲染公共游戏检查点，并把全部公共自由文本封入数据引用。"""

    game = snapshot.game
    source_references = (
        *game.source_references,
        *(entry.source_reference for entry in game.public_summary),
    )
    public_lines: list[str] = []
    for entry in sorted(game.public_summary, key=lambda item: item.entry_id):
        public_lines.append(f"> entry_id: {entry.entry_id}")
        public_lines.extend(_quoted_untrusted_lines(entry.text))
    for index, text in enumerate(sorted(snapshot.bounded_public_summary), start=1):
        public_lines.append(f"> summary_index: {index}")
        public_lines.extend(_quoted_untrusted_lines(text))
    private_refs = sorted(
        (*game.authorized_private_fact_references, *snapshot.critical_private_fact_references),
        key=lambda item: (item.record_id, item.revision, item.content_hash),
    )
    content = _canonical_document("GAME.md", [
        "## CHECKPOINT",
        f"- game_id: {snapshot.game_id}",
        f"- base_game_revision: {snapshot.base_game_revision}",
        f"- view_fingerprint: {snapshot.view_fingerprint}",
        f"- day: {game.day}",
        f"- phase: {game.phase}",
        "## LIVING_PLAYERS",
        *(f"- {player_id}" for player_id in sorted(game.living_player_ids)),
        "## AUTHORIZED_PRIVATE_FACT_REFERENCES",
        *(
            f"- {reference.record_id}@{reference.revision} sha256:{reference.content_hash}"
            for reference in private_refs
        ),
        "## SOURCES",
        *_source_lines(source_references),
        "## UNTRUSTED_PUBLIC_DATA",
        *public_lines,
    ])
    return _build_document(
        section_id=WorkspaceSection.GAME,
        renderer_version=GAME_RENDERER_VERSION,
        identity=snapshot.identity,
        content_markdown=content,
        estimator=estimator,
        visibility_class=ProjectionVisibilityClass.PUBLIC,
        source_references=source_references,
    )


def _move_detail_lines(move: SpeechMove) -> list[str]:
    """按动作类型渲染结构化字段，绝不读取任何口语文本。"""

    lines = [
        f"  - move_type: {move.move_type}",
        f"  - modality: {move.modality}",
        f"  - evidence_refs: {','.join(sorted(move.evidence_refs)) or '-'}",
    ]
    if isinstance(move, AlignmentRead):
        lines.extend((f"  - target_id: {move.target_id}", f"  - alignment: {move.alignment}", f"  - strength: {move.strength}"))
    elif isinstance(move, RoleClaim):
        lines.extend((f"  - claimant_id: {move.claimant_id}", f"  - role_id: {move.role_id}", f"  - claim_mode: {move.claim_mode}", f"  - source_record_id: {move.source_record_id or '-'}"))
    elif isinstance(move, PrivateResultDisclosure):
        lines.extend((f"  - fact_kind: {move.fact_kind}", f"  - fact_ref: {move.fact_ref}", f"  - disclosure_grant_id: {move.disclosure_grant_id}", f"  - timing_ref: {move.timing_ref}", f"  - result_value_id: {move.result_value_id}", f"  - target_id: {move.target_id or '-'}"))
    elif isinstance(move, PublicEvidenceCitation):
        lines.extend((f"  - relation: {move.relation}", f"  - subject_ids: {','.join(sorted(move.subject_ids))}", f"  - supports_move_ids: {','.join(sorted(move.supports_move_ids))}"))
    elif isinstance(move, PlayerComparison):
        lines.append(f"  - dimension: {move.dimension}")
        lines.extend(f"  - assessment: {item.player_id}/{item.value_id}/{','.join(sorted(item.evidence_refs))}" for item in sorted(move.assessments, key=lambda item: item.player_id))
    elif isinstance(move, QuestionMove):
        lines.extend((f"  - target_id: {move.target_id}", f"  - topic: {move.topic}", f"  - requested_fields: {','.join(sorted(move.requested_fields))}"))
    elif isinstance(move, ResponseMove):
        lines.extend((f"  - source_record_id: {move.source_record_id}", f"  - response_kind: {move.response_kind}"))
    elif isinstance(move, VotePosition):
        lines.extend((f"  - target_id: {move.target_id}", f"  - commitment: {move.commitment}"))
    elif isinstance(move, ConditionalCommitment):
        lines.extend((f"  - condition_id: {move.condition.condition_id}", f"  - condition_kind: {move.condition.kind_id}", f"  - condition_record_refs: {','.join(sorted(move.condition.record_refs)) or '-'}", f"  - consequence_kind: {move.consequence.kind}", f"  - consequence_target_id: {move.consequence.target_id or '-'}", f"  - expires_at_phase: {move.expires_at_phase}"))
    elif isinstance(move, RetractionMove):
        lines.extend((f"  - prior_public_move_ref: {move.prior_public_move_ref}", f"  - replacement_move_id: {move.replacement_move_id or '-'}"))
    elif isinstance(move, UncertaintyStatement):
        lines.extend((f"  - subject_id: {move.subject_id}", f"  - dimension: {move.dimension}"))
        lines.extend(f"  - alternative: {item.value_id}/{item.confidence}/{','.join(sorted(item.support_refs)) or '-'}" for item in sorted(move.alternatives, key=lambda item: item.value_id))
    return lines


def _commitment_order(source: CommitmentProjectionSource) -> tuple[int, str, int, str]:
    """按提交修订与来源身份固定承诺文档顺序。"""

    return (
        source.record.committed_revision,
        source.record.record_id,
        source.source_reference.record_revision,
        source.source_reference.content_hash,
    )


def render_commitments_document(
    snapshot: ObservationAuthoritySnapshot,
    estimator: TokenEstimator,
) -> ProjectedDocument:
    """渲染当前玩家已提交的结构化承诺，不使用渲染口语作为权威。"""

    if snapshot.commitment_records is None:
        raise ValueError("commitment source capability is unavailable")
    commitments = tuple(sorted(snapshot.commitment_records, key=_commitment_order))
    lines = ["## COMMITTED_RECORDS"]
    for source in commitments:
        record = source.record
        if record.actor_id != snapshot.player_id or record.game_id != snapshot.game_id:
            raise ValueError("commitment record does not belong to snapshot viewer")
        lines.extend((
            f"- record_id: {record.record_id}",
            f"  - actor_id: {record.actor_id}",
            f"  - game_id: {record.game_id}",
            f"  - day: {record.day}",
            f"  - phase: {record.phase}",
            f"  - committed_revision: {record.committed_revision}",
            f"  - source_hash: {source.source_reference.content_hash}",
            f"  - source_evidence_refs: {','.join(sorted(record.source_evidence_refs)) or '-'}",
            f"  - disclosure_grant_refs: {','.join(sorted(record.disclosure_grant_refs)) or '-'}",
        ))
        for move in sorted(record.normalized_moves, key=lambda item: item.move_id):
            lines.append(f"  - move_id: {move.move_id}")
            lines.extend(_move_detail_lines(move))
    source_references = tuple(source.source_reference for source in commitments)
    content = _canonical_document("COMMITMENTS.md", [
        *lines,
        "## SOURCES",
        *_source_lines(source_references),
    ])
    return _build_document(
        section_id=WorkspaceSection.COMMITMENTS,
        renderer_version=COMMITMENTS_RENDERER_VERSION,
        identity=snapshot.identity,
        content_markdown=content,
        estimator=estimator,
        visibility_class=ProjectionVisibilityClass.MIXED_VIEWER_FILTERED,
        source_references=source_references,
    )


class _DedicatedRenderer:
    """把固定区段元数据绑定到专用渲染函数。"""

    def __init__(
        self,
        *,
        section_id: WorkspaceSection,
        renderer_version: str,
        render_function: Callable[
            [ObservationAuthoritySnapshot, TokenEstimator], ProjectedDocument
        ],
    ) -> None:
        self.section_id = section_id
        self.renderer_version = renderer_version
        self._render_function = render_function

    def render(
        self,
        snapshot: ObservationAuthoritySnapshot,
        estimator: TokenEstimator,
    ) -> ProjectedDocument:
        return self._render_function(snapshot, estimator)


DOCUMENT_RENDERERS: Mapping[WorkspaceSection, DocumentRenderer] = {
    WorkspaceSection.PLAYER: _DedicatedRenderer(
        section_id=WorkspaceSection.PLAYER,
        renderer_version=PLAYER_RENDERER_VERSION,
        render_function=render_player_document,
    ),
    WorkspaceSection.ROLE: _DedicatedRenderer(
        section_id=WorkspaceSection.ROLE,
        renderer_version=ROLE_RENDERER_VERSION,
        render_function=render_role_document,
    ),
    WorkspaceSection.GAME: _DedicatedRenderer(
        section_id=WorkspaceSection.GAME,
        renderer_version=GAME_RENDERER_VERSION,
        render_function=render_game_document,
    ),
    WorkspaceSection.COMMITMENTS: _DedicatedRenderer(
        section_id=WorkspaceSection.COMMITMENTS,
        renderer_version=COMMITMENTS_RENDERER_VERSION,
        render_function=render_commitments_document,
    ),
}


__all__ = [
    "COMMITMENTS_RENDERER_VERSION",
    "DOCUMENT_RENDERERS",
    "GAME_RENDERER_VERSION",
    "PLAYER_RENDERER_VERSION",
    "ROLE_RENDERER_VERSION",
    "ConservativeTokenEstimator",
    "DocumentRenderer",
    "TokenEstimator",
    "render_commitments_document",
    "render_game_document",
    "render_player_document",
    "render_role_document",
]
