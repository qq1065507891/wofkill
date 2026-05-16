"""RAG schemas: case metadata, quality grades, retrieval hits.

Every RAG entry has mandatory source metadata, quality grading, and
visibility boundary. Forbidden content types are enforced at ingestion.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CaseType(str, Enum):
    ROLE_STRATEGY = "role_strategy"
    SPEECH_TEMPLATE = "speech_template"
    EXTERNAL_HIGH_END_CASE = "external_high_end_case"
    EXTERNAL_TACTICS = "external_tactics"
    PROJECT_HISTORY = "project_history"
    PROJECT_REVIEW = "project_review"


class QualityGrade(str, Enum):
    PRO_MATCH = "pro_match"
    EXPERT_REVIEW = "expert_review"
    HIGH_RANK_GAME = "high_rank_game"
    COMMUNITY_CASE = "community_case"
    SELF_PLAY_CANDIDATE = "self_play_candidate"
    RULE_DERIVED_SEED = "rule_derived_seed"
    UNREVIEWED = "unreviewed"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class VisibilityBoundary(str, Enum):
    PUBLIC_ONLY = "public_only"
    PLAYER_PERSPECTIVE = "player_perspective"
    GOD_VIEW = "god_view"
    MODERATOR_ONLY = "moderator_only"


class SourceType(str, Enum):
    PUBLIC_TOURNAMENT = "public_tournament"
    PUBLIC_REVIEW = "public_review"
    EXPERT_COMMENTARY = "expert_commentary"
    TRAINING_SESSION = "training_session"
    SELF_PLAY = "self_play"
    RULE_DERIVED = "rule_derived"
    MANUAL_ENTRY = "manual_entry"


# ---------------------------------------------------------------------------
# Forbidden content keywords — RAG must never contain these
# ---------------------------------------------------------------------------

FORBIDDEN_RAG_CONTENT_TYPES: set[str] = {
    "base_rule",
    "role_skill_truth",
    "victory_truth",
    "adjudication_truth",
    "live_game_state",
}

FORBIDDEN_RAG_KEYWORDS: set[str] = {
    "rule_engine_says",
    "actual_role_is",
    "ground_truth_alignment",
    "moderator_knows",
}


# ---------------------------------------------------------------------------
# Case metadata
# ---------------------------------------------------------------------------

class SourceMetadata(BaseModel):
    """Source attribution for a RAG entry."""
    source_type: SourceType
    source_url: str = ""
    source_title: str = ""
    source_author: str = ""
    publish_date: str = ""
    collected_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class CaseMetadata(BaseModel):
    """Full metadata for a RAG case entry."""
    case_type: CaseType
    quality_grade: QualityGrade = QualityGrade.UNREVIEWED
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer: str = ""
    ruleset_id: str = ""
    player_count: int = 12
    phase: str = ""
    role_perspective: str = ""
    visibility_boundary: VisibilityBoundary = VisibilityBoundary.PLAYER_PERSPECTIVE
    source: SourceMetadata
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG entry — the full case stored in the knowledge base
# ---------------------------------------------------------------------------

class RAGEntry(BaseModel):
    """A single RAG knowledge base entry."""
    entry_id: str
    title: str
    summary: str = Field(..., description="Structured summary, not full-text dump")
    key_decisions: list[str] = Field(default_factory=list)
    short_quotes: list[str] = Field(default_factory=list, description="Brief quotes, not full paragraphs")
    metadata: CaseMetadata
    content_type: str = Field("strategy", description="What kind of content this contains")

    @field_validator("content_type")
    @classmethod
    def content_type_not_forbidden(cls, v: str) -> str:
        if v in FORBIDDEN_RAG_CONTENT_TYPES:
            raise ValueError(f"RAG entry content_type '{v}' is forbidden")
        return v


# ---------------------------------------------------------------------------
# RAG hit — a retrieval result
# ---------------------------------------------------------------------------

class RAGHit(BaseModel):
    """A single RAG retrieval result, annotated for display."""
    entry_id: str
    title: str
    summary: str
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    quality_grade: QualityGrade
    source_type: SourceType
    visibility_boundary: VisibilityBoundary
    case_type: CaseType
    role_perspective: str = ""
    phase: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    short_quotes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    allowed_in_live_context: bool = True
    display_annotation: str = ""

    @model_validator(mode="after")
    def compute_allowed_in_live(self) -> "RAGHit":
        if self.visibility_boundary in (
            VisibilityBoundary.GOD_VIEW,
            VisibilityBoundary.MODERATOR_ONLY,
        ):
            self.allowed_in_live_context = False
        return self


# ---------------------------------------------------------------------------
# RAG retrieval query
# ---------------------------------------------------------------------------

class RAGQuery(BaseModel):
    """A retrieval query for the strategy retriever."""
    role: str = ""
    phase: str = ""
    situation: str = ""
    ruleset_id: str = ""
    persona_style: str = ""
    quality_min: QualityGrade | None = None
    source_types: list[SourceType] = Field(default_factory=list)
    case_types: list[CaseType] = Field(default_factory=list)
    max_results: int = 5
    include_god_view: bool = False
    viewer_role: str = ""
