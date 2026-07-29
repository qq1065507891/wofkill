# -*- coding: utf-8 -*-
"""
自主玩家智能体的严格、版本化且无副作用的公共契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from werewolf_agent.player_agents.contracts.disclosure import DisclosureGrant
from werewolf_agent.player_agents.contracts.errors import (
    ProposalFailure,
    ValidationErrorCode,
)
from werewolf_agent.player_agents.contracts.proposals import SpeechProposalEnvelope
from werewolf_agent.player_agents.contracts.records import (
    PublicSpeechRecord,
    RecordOrigin,
    RenderedUtterance,
)
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.speech import (
    AddressStyle,
    Alignment,
    AlignmentRead,
    ClaimMode,
    ComparisonDimension,
    ConditionalCommitment,
    ConfidenceBucket,
    ConsequenceKind,
    EvidenceRelation,
    LengthClass,
    Modality,
    PlayerComparison,
    PrivateFactKind,
    PrivateResultDisclosure,
    PublicEvidenceCitation,
    QuestionMove,
    QuestionTopic,
    RequestedField,
    ResponseKind,
    ResponseMove,
    RetractionMove,
    RoleClaim,
    SpeechMove,
    SpeechObjective,
    SpeechProposalBody,
    Strength,
    Tone,
    UncertaintyDimension,
    UncertaintyStatement,
    VoteCommitment,
    VotePosition,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
    transition_turn,
)

__all__ = [
    "AddressStyle",
    "AgentTurn",
    "AgentTurnStatus",
    "Alignment",
    "AlignmentRead",
    "ClaimMode",
    "ComparisonDimension",
    "ConditionalCommitment",
    "ConfidenceBucket",
    "ConflictClass",
    "ConsequenceKind",
    "DisclosureGrant",
    "EvidenceRelation",
    "LegalActionWindow",
    "LengthClass",
    "Modality",
    "PlayerComparison",
    "PrivateFactKind",
    "PrivateResultDisclosure",
    "ProposalFailure",
    "PublicEvidenceCitation",
    "PublicSpeechRecord",
    "QuestionMove",
    "QuestionTopic",
    "ReadReference",
    "RecordOrigin",
    "RenderedUtterance",
    "RequestedField",
    "ResponseKind",
    "ResponseMove",
    "RetractionMove",
    "RevisionContext",
    "RoleClaim",
    "SpeechMove",
    "SpeechObjective",
    "SpeechProposalBody",
    "SpeechProposalEnvelope",
    "Strength",
    "Tone",
    "TurnBudget",
    "UncertaintyDimension",
    "UncertaintyStatement",
    "ValidationErrorCode",
    "VoteCommitment",
    "VotePosition",
    "transition_turn",
]
