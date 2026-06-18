"""Memory schemas: structured types for short-term, long-term, and review memory.

Design doc §10: short-term uses JSON cognition matrix (not vectors).
Vote chains, claims, attack/defense relations stay as structured data.
Vector storage is reserved for unstructured reflections only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation graph predicates
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    SPOKE_AGAINST = "spoke_against"
    VOTED = "voted"
    CLAIMED_ROLE = "claimed_role"
    DEFENDED = "defended"
    NIGHT_RESULT_CLAIMED = "night_result_claimed"


@dataclass(frozen=True)
class RelationEvent:
    predicate: RelationType
    source: str
    target: str | None = None
    day: int = 0
    value: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cognition matrix entry — per-player short-term state
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """Structured evidence reference carried in CognitionMatrixEntry.key_evidence.

    MEM-07: a bare ``str`` evidence entry has no provenance, claim,
    or confidence — making it impossible to debug, filter, or weight
    observations downstream. The structured form carries:

    * ``claim`` — the human-readable assertion (e.g. "p03 is wolf")
    * ``source_event`` — the GameEvent.type that produced the claim
    * ``day`` — the game day the claim originated on
    * ``confidence`` — the producer's confidence in [0.0, 1.0]
    * ``speaker`` — optional actor (player id) the claim is about

    The dataclass is plain (not frozen) to allow in-place edits; use
    ``to_dict`` / ``from_dict`` for serialization.
    """
    claim: str
    source_event: str = ""
    day: int = 0
    confidence: float = 0.5
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "source_event": self.source_event,
            "day": self.day,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        return cls(
            claim=data.get("claim", str(data)),
            source_event=data.get("source_event", ""),
            day=data.get("day", 0),
            confidence=data.get("confidence", 0.5),
            speaker=data.get("speaker"),
        )


@dataclass
class CognitionMatrixEntry:
    player_id: str
    role_probabilities: dict[str, float] = field(default_factory=dict)
    faction_read: str = "unknown"
    trust: float = 0.5
    key_evidence: list[Any] = field(default_factory=list)  # EvidenceItem (or str for back-compat)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "role_probabilities": dict(self.role_probabilities),
            "faction_read": self.faction_read,
            "trust": self.trust,
            "key_evidence": [
                e.to_dict() if isinstance(e, EvidenceItem) else e
                for e in self.key_evidence
            ],
            "open_questions": list(self.open_questions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitionMatrixEntry":
        evidence = []
        for e in data.get("key_evidence", []):
            if isinstance(e, dict):
                evidence.append(EvidenceItem.from_dict(e))
            else:
                evidence.append(e)  # back-compat: keep bare str
        return cls(
            player_id=data["player_id"],
            role_probabilities=data.get("role_probabilities", {}),
            faction_read=data.get("faction_read", "unknown"),
            trust=data.get("trust", 0.5),
            key_evidence=evidence,
            open_questions=data.get("open_questions", []),
        )


# ---------------------------------------------------------------------------
# Player profile — ability scores with growth tracking
# ---------------------------------------------------------------------------

@dataclass
class PlayerProfile:
    player_id: str
    logic: float = 0.5
    deception: float = 0.5
    leadership: float = 0.5
    credibility: float = 0.5
    learning_rate: float = 0.1
    risk_preference: float = 0.5
    games_played: int = 0
    games_as_wolf: int = 0
    games_as_good: int = 0
    wolf_wins: int = 0
    good_wins: int = 0
    review_history: list[str] = field(default_factory=list)

    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wolf_wins + self.good_wins) / self.games_played

    def apply_deltas(self, deltas: dict[str, float]) -> None:
        valid = ("logic", "deception", "leadership", "credibility",
                 "learning_rate", "risk_preference")
        for k, v in deltas.items():
            if k in valid:
                new_val = getattr(self, k) + v
                setattr(self, k, max(0.0, min(1.0, new_val)))
            else:
                # MEM-09: silently dropping typo'd deltas hides bugs
                # in the upstream review generator. Log a warning
                # so the caller can spot the misspelling.
                _LOG.warning("Unknown delta attr %s ignored", k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "logic": self.logic,
            "deception": self.deception,
            "leadership": self.leadership,
            "credibility": self.credibility,
            "learning_rate": self.learning_rate,
            "risk_preference": self.risk_preference,
            "games_played": self.games_played,
            "games_as_wolf": self.games_as_wolf,
            "games_as_good": self.games_as_good,
            "wolf_wins": self.wolf_wins,
            "good_wins": self.good_wins,
            "review_history": list(self.review_history),
        }


# ---------------------------------------------------------------------------
# Review — post-game analysis per player
# ---------------------------------------------------------------------------

@dataclass
class ReviewJudgment:
    target_player: str
    judgment: str  # "correct" or "incorrect"
    actual_role: str
    guessed_role: str
    evidence: str = ""
    day: int = 0


@dataclass
class ReviewReport:
    game_id: str
    player_id: str
    role: str
    faction_won: bool
    key_judgments: list[ReviewJudgment] = field(default_factory=list)
    error_analysis: list[str] = field(default_factory=list)
    successful_strategies: list[str] = field(default_factory=list)
    deceived_by: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    ability_deltas: dict[str, float] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# Reflection V2 — structured long-term learning cards
# ---------------------------------------------------------------------------


class ReflectionQualityStatus(str, Enum):
    APPROVED = "approved"
    REVIEW_ONLY = "review_only"
    REJECTED = "rejected"


class SituationSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = ""
    faction: str = ""
    outcome: str = ""
    phase_focus: list[str] = Field(default_factory=list)
    game_patterns: list[str] = Field(default_factory=list)


class ReflectionMistakePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = ""
    trigger: str = ""
    wrong_action: str = ""
    better_action: str = ""
    fact_basis: str = "llm_transferable"
    auto_verified: bool = False
    corrected_from_llm: bool = False


class ReflectionPreservedStrength(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = ""
    behavior: str = ""
    reuse_condition: str = ""


class ReflectionPromptCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str
    lesson: str
    trigger_signals: list[str] = Field(default_factory=list)
    recommended_action: str
    misuse_risk: str
    fact_basis: str = "llm_transferable"
    auto_verified: bool = False


class ReflectionSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    llm_self_review: str = ""
    auto_review_summary: str = ""
    merged_by: str = "reflection_synthesizer_v2"


class ReflectionEntryV2(BaseModel):
    """Structured reflection entry stored inside reflections.entry_json.

    Only ``prompt_card`` and aggregate labels derived from approved V2 entries
    may be rendered into live prompts. Source and quality fields are audit-only.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    entry_id: str
    game_id: str
    player_id: str
    role: str
    faction: str = ""
    faction_won: bool = False
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_status: ReflectionQualityStatus = ReflectionQualityStatus.REVIEW_ONLY
    quality_flags: list[str] = Field(default_factory=list)
    situation_signature: SituationSignature = Field(default_factory=SituationSignature)
    mistake_patterns: list[ReflectionMistakePattern] = Field(default_factory=list)
    preserved_strengths: list[ReflectionPreservedStrength] = Field(default_factory=list)
    actionable_advice: list[str] = Field(default_factory=list)
    avoid_next_time: list[str] = Field(default_factory=list)
    prompt_card: ReflectionPromptCard
    source: ReflectionSource = Field(default_factory=ReflectionSource)

    @model_validator(mode="after")
    def _identity_must_exist(self) -> "ReflectionEntryV2":
        if not self.entry_id.strip():
            raise ValueError("ReflectionEntryV2.entry_id must be non-empty")
        if not self.game_id.strip():
            raise ValueError("ReflectionEntryV2.game_id must be non-empty")
        if not self.player_id.strip():
            raise ValueError("ReflectionEntryV2.player_id must be non-empty")
        if not self.role.strip():
            raise ValueError("ReflectionEntryV2.role must be non-empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionEntryV2":
        return cls.model_validate(data)

    def prompt_visible_texts(self) -> list[str]:
        """Return every field that can influence live prompt rendering."""
        texts = [
            self.prompt_card.theme,
            self.prompt_card.lesson,
            self.prompt_card.recommended_action,
            self.prompt_card.misuse_risk,
            *self.prompt_card.trigger_signals,
            *self.actionable_advice,
            *self.avoid_next_time,
        ]
        for pattern in self.mistake_patterns:
            # wrong_action carries deterministic ground-truth (review.py:108
            # writes "实际 {actual}"); it is audit-only, not prompt-visible
            # (spec: prompt_card is the only player-prompt-facing summary).
            texts.extend([
                pattern.category,
                pattern.trigger,
                pattern.better_action,
            ])
        for strength in self.preserved_strengths:
            texts.extend([
                strength.category,
                strength.behavior,
                strength.reuse_condition,
            ])
        return [str(t) for t in texts if str(t or "").strip()]


# ---------------------------------------------------------------------------
# Reflection — long-term unstructured experience
# ---------------------------------------------------------------------------

# MEM-28: caps on the tags field of ReflectionEntry. A 1000-tag
# entry serializes to a multi-KB JSON blob per reflection, and the
# cost compounds in the vector index's per-entry IDF bookkeeping.
# 20 tags × 64 chars is plenty for any meaningful reflection label
# (role, win/loss, error class, strategy class) without bloating
# storage.
_MAX_REFLECTION_TAGS = 20
_MAX_REFLECTION_TAG_LEN = 64


def _validate_reflection_tags(tags: list[str]) -> None:
    """MEM-28: enforce the tag-count and per-tag-length caps.

    Raises ValueError with a descriptive message so the caller can
    see which limit was violated. Empty tag strings are allowed
    (callers may strip / normalize upstream), but very long strings
    and very large lists are not.
    """
    if len(tags) > _MAX_REFLECTION_TAGS:
        raise ValueError(
            f"ReflectionEntry.tags accepts at most "
            f"{_MAX_REFLECTION_TAGS} tags, got {len(tags)}"
        )
    for i, tag in enumerate(tags):
        if not isinstance(tag, str):
            raise ValueError(
                f"ReflectionEntry.tags[{i}] must be a string, "
                f"got {type(tag).__name__}"
            )
        if len(tag) > _MAX_REFLECTION_TAG_LEN:
            raise ValueError(
                f"ReflectionEntry.tags[{i}] is {len(tag)} chars, "
                f"max is {_MAX_REFLECTION_TAG_LEN}"
            )


@dataclass
class ReflectionEntry:
    entry_id: str
    game_id: str
    player_id: str
    role: str
    faction_won: bool
    text: str
    tags: list[str] = field(default_factory=list)
    situation: str = ""

    def __post_init__(self) -> None:
        # MEM-28: bound the tags list to prevent JSON blow-up.
        # Run after the dataclass __init__ so ``self.tags`` is the
        # final list (or factory default).
        _validate_reflection_tags(self.tags)
        # MEM-NEW-10: reject empty game_id at construction time. A
        # reflection with ``game_id=""`` is silently invisible in
        # the chr-invert newest-first sort — the empty string
        # always sorts to the end, so the entry is effectively
        # unreachable for cross-game queries. Surface the bug at
        # construction rather than letting it rot in the index.
        if not self.game_id or not self.game_id.strip():
            raise ValueError(
                f"ReflectionEntry.game_id must be a non-empty string; "
                f"got {self.game_id!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "game_id": self.game_id,
            "player_id": self.player_id,
            "role": self.role,
            "faction_won": self.faction_won,
            "text": self.text,
            "tags": list(self.tags),
            "situation": self.situation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionEntry":
        return cls(
            entry_id=data["entry_id"],
            game_id=data.get("game_id", ""),
            player_id=data.get("player_id", ""),
            role=data.get("role", ""),
            faction_won=data.get("faction_won", False),
            text=data.get("text", ""),
            tags=data.get("tags", []),
            situation=data.get("situation", ""),
        )


# ---------------------------------------------------------------------------
# Cross-game query
# ---------------------------------------------------------------------------

@dataclass
class CrossGameQuery:
    player_id: str = ""
    role: str = ""
    # OR semantics: entry matches if any tag in this list is in
    # entry.tags. CrossGameQuery combines this with player_id/role/
    # faction_won equality, all of which are AND-joined. Within
    # ``tags`` itself, the match is OR (i.e. union).
    tags: list[str] = field(default_factory=list)
    situation: str = ""
    max_results: int = 5
    faction_won: bool | None = None
