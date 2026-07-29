# -*- coding: utf-8 -*-
"""
定义昼间发言的严格语义动作联合、引用关系和交付计划。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-29
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)


class Modality(StrEnum):
    ASSERTED = "asserted"
    SUSPECTED = "suspected"
    CONDITIONAL = "conditional"
    HYPOTHETICAL = "hypothetical"
    QUOTED = "quoted"


class SpeechObjective(StrEnum):
    STATE_CASE = "state_case"
    CHALLENGE_CLAIM = "challenge_claim"
    ANSWER_QUESTION = "answer_question"
    ASK_QUESTION = "ask_question"
    DEFEND_SELF = "defend_self"
    DECLARE_VOTE_POSITION = "declare_vote_position"
    RETRACT_OR_CORRECT = "retract_or_correct"
    EXPRESS_UNCERTAINTY = "express_uncertainty"
    NO_NEW_INFORMATION = "no_new_information"


class Alignment(StrEnum):
    GOOD = "good"
    WOLF = "wolf"
    UNCERTAIN = "uncertain"


class Strength(StrEnum):
    LEANING = "leaning"
    PROBABLE = "probable"
    COMMITTED = "committed"


class ClaimMode(StrEnum):
    CLAIM = "claim"
    DENY = "deny"
    QUOTE = "quote"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"


class ComparisonDimension(StrEnum):
    CLAIM = "claim"
    VOTE = "vote"
    EVIDENCE = "evidence"
    COMMITMENT = "commitment"
    TIMELINE_CONSISTENCY = "timeline_consistency"


class QuestionTopic(StrEnum):
    ROLE_CLAIM = "role_claim"
    ALIGNMENT_READ = "alignment_read"
    VOTE_POSITION = "vote_position"
    EVIDENCE = "evidence"
    TIMELINE = "timeline"
    COMMITMENT = "commitment"


class RequestedField(StrEnum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    REASON = "reason"
    TIMELINE = "timeline"
    VOTE = "vote"
    CONFIDENCE = "confidence"


class ResponseKind(StrEnum):
    AGREE = "agree"
    DISAGREE = "disagree"
    CLARIFY = "clarify"
    CHALLENGE = "challenge"


class VoteCommitment(StrEnum):
    LEANING = "leaning"
    PROVISIONAL = "provisional"
    COMMITTED = "committed"


class ConsequenceKind(StrEnum):
    SUPPORT = "support"
    CHALLENGE = "challenge"
    VOTE_POSITION = "vote_position"
    SELF_DISCLOSURE = "self_disclosure"


class UncertaintyDimension(StrEnum):
    ROLE = "role"
    ALIGNMENT = "alignment"
    VOTE = "vote"
    CLAIM = "claim"
    TIMELINE = "timeline"


class ConfidenceBucket(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Tone(StrEnum):
    CALM = "calm"
    FIRM = "firm"
    SKEPTICAL = "skeptical"
    URGENT = "urgent"
    DEFENSIVE = "defensive"
    CONCILIATORY = "conciliatory"


class LengthClass(StrEnum):
    BRIEF = "brief"
    STANDARD = "standard"
    EXTENDED = "extended"


class AddressStyle(StrEnum):
    ROOM = "room"
    TARGETED = "targeted"
    MIXED = "mixed"


class PrivateFactKind(StrEnum):
    ALIGNMENT_CHECK = "alignment_check"
    ROLE_CHECK = "role_check"
    ATTACK = "attack"
    ABILITY_RESULT = "ability_result"


class BaseMove(StrictFrozenModel):
    move_id: NonEmptyId
    move_type: NonEmptyId
    modality: Modality
    evidence_refs: tuple[NonEmptyId, ...] = ()

    @field_validator("evidence_refs")
    @classmethod
    def _unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(value, field_name="evidence_refs")


class AlignmentRead(BaseMove):
    move_type: Literal["alignment_read"]
    target_id: NonEmptyId
    alignment: Alignment
    strength: Strength


class RoleClaim(BaseMove):
    move_type: Literal["role_claim"]
    claimant_id: NonEmptyId
    role_id: NonEmptyId
    claim_mode: ClaimMode
    source_record_id: NonEmptyId | None = None

    @model_validator(mode="after")
    def _quoted_claim_has_source(self) -> RoleClaim:
        if (self.claim_mode is ClaimMode.QUOTE) != (self.source_record_id is not None):
            raise ValueError("quoted role claim requires exactly one source record")
        return self


class PrivateResultDisclosure(BaseMove):
    move_type: Literal["private_result_disclosure"]
    fact_kind: PrivateFactKind
    fact_ref: NonEmptyId
    disclosure_grant_id: NonEmptyId
    timing_ref: NonEmptyId
    result_value_id: NonEmptyId
    target_id: NonEmptyId | None = None


class PublicEvidenceCitation(BaseMove):
    move_type: Literal["public_evidence_citation"]
    relation: EvidenceRelation
    subject_ids: tuple[NonEmptyId, ...] = Field(min_length=1, max_length=4)
    supports_move_ids: tuple[NonEmptyId, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _unique_citation_refs(self) -> PublicEvidenceCitation:
        require_unique(self.subject_ids, field_name="subject_ids")
        require_unique(self.supports_move_ids, field_name="supports_move_ids")
        return self


class ComparisonAssessment(StrictFrozenModel):
    player_id: NonEmptyId
    value_id: NonEmptyId
    evidence_refs: tuple[NonEmptyId, ...] = Field(min_length=1)


class PlayerComparison(BaseMove):
    move_type: Literal["player_comparison"]
    dimension: ComparisonDimension
    assessments: tuple[ComparisonAssessment, ...] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def _unique_players(self) -> PlayerComparison:
        require_unique(
            (item.player_id for item in self.assessments),
            field_name="comparison players",
        )
        return self


class QuestionMove(BaseMove):
    move_type: Literal["question"]
    target_id: NonEmptyId
    topic: QuestionTopic
    requested_fields: tuple[RequestedField, ...] = Field(min_length=1)

    @field_validator("requested_fields")
    @classmethod
    def _unique_requested_fields(
        cls,
        value: tuple[RequestedField, ...],
    ) -> tuple[RequestedField, ...]:
        return require_unique(value, field_name="requested_fields")


class ResponseMove(BaseMove):
    move_type: Literal["response"]
    source_record_id: NonEmptyId
    response_kind: ResponseKind


class VotePosition(BaseMove):
    move_type: Literal["vote_position"]
    target_id: NonEmptyId
    commitment: VoteCommitment


class CommitmentCondition(StrictFrozenModel):
    condition_id: NonEmptyId
    kind_id: NonEmptyId
    record_refs: tuple[NonEmptyId, ...] = ()

    @field_validator("record_refs")
    @classmethod
    def _unique_records(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(value, field_name="record_refs")


class CommitmentConsequence(StrictFrozenModel):
    kind: ConsequenceKind
    target_id: NonEmptyId | None = None


class ConditionalCommitment(BaseMove):
    move_type: Literal["conditional_commitment"]
    condition: CommitmentCondition
    consequence: CommitmentConsequence
    expires_at_phase: NonEmptyId


class RetractionMove(BaseMove):
    move_type: Literal["retraction"]
    prior_public_move_ref: NonEmptyId
    replacement_move_id: NonEmptyId | None = None


class UncertaintyAlternative(StrictFrozenModel):
    value_id: NonEmptyId
    confidence: ConfidenceBucket
    support_refs: tuple[NonEmptyId, ...] = ()


class UncertaintyStatement(BaseMove):
    move_type: Literal["uncertainty"]
    subject_id: NonEmptyId
    dimension: UncertaintyDimension
    alternatives: tuple[UncertaintyAlternative, ...] = Field(min_length=2, max_length=4)


SpeechMove = Annotated[
    AlignmentRead
    | RoleClaim
    | PrivateResultDisclosure
    | PublicEvidenceCitation
    | PlayerComparison
    | QuestionMove
    | ResponseMove
    | VotePosition
    | ConditionalCommitment
    | RetractionMove
    | UncertaintyStatement,
    Field(discriminator="move_type"),
]


def _require_acyclic(graph: dict[str, set[str]]) -> None:
    unseen, visiting, complete = 0, 1, 2
    colors = {node: unseen for node in graph}

    def visit(node: str) -> None:
        if colors[node] == visiting:
            raise ValueError("move references must be acyclic")
        if colors[node] == complete:
            return
        colors[node] = visiting
        for target in sorted(graph[node]):
            visit(target)
        colors[node] = complete

    for node in graph:
        visit(node)


class DeliveryPlan(StrictFrozenModel):
    tone: Tone
    length_class: LengthClass
    address_style: AddressStyle
    move_order: tuple[NonEmptyId, ...]
    emphasis_move_ids: tuple[NonEmptyId, ...] = ()
    connector_ids: tuple[NonEmptyId, ...] = ()


class SpeechProposalBody(StrictFrozenModel):
    kind: Literal["speech"]
    objective: SpeechObjective
    moves: tuple[SpeechMove, ...] = Field(min_length=1, max_length=8)
    response_record_refs: tuple[NonEmptyId, ...] = ()
    delivery_plan: DeliveryPlan

    @model_validator(mode="after")
    def _validate_move_graph(self) -> SpeechProposalBody:
        move_ids = tuple(move.move_id for move in self.moves)
        require_unique(move_ids, field_name="move IDs")
        if set(self.delivery_plan.move_order) != set(move_ids) or len(
            self.delivery_plan.move_order
        ) != len(move_ids):
            raise ValueError("move_order must contain every move ID exactly once")
        require_unique(
            self.delivery_plan.emphasis_move_ids,
            field_name="emphasis_move_ids",
        )
        if not set(self.delivery_plan.emphasis_move_ids) <= set(move_ids):
            raise ValueError("emphasis_move_ids must reference proposal moves")

        require_unique(
            self.response_record_refs,
            field_name="response_record_refs",
        )
        external_refs = {
            move.source_record_id
            for move in self.moves
            if isinstance(move, ResponseMove)
        } | {
            move.source_record_id
            for move in self.moves
            if isinstance(move, RoleClaim) and move.source_record_id is not None
        }
        if set(self.response_record_refs) != external_refs:
            raise ValueError("response_record_refs must match referenced records")

        graph: dict[str, set[str]] = {move_id: set() for move_id in move_ids}
        for move in self.moves:
            if isinstance(move, PublicEvidenceCitation):
                graph[move.move_id].update(move.supports_move_ids)
            if isinstance(move, RetractionMove) and move.replacement_move_id:
                graph[move.move_id].add(move.replacement_move_id)
        if any(target not in graph for targets in graph.values() for target in targets):
            raise ValueError("move reference must resolve inside the proposal")
        _require_acyclic(graph)
        return self
