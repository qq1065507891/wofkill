"""Memory schemas: structured types for short-term, long-term, and review memory.

Design doc §10: short-term uses JSON cognition matrix (not vectors).
Vote chains, claims, attack/defense relations stay as structured data.
Vector storage is reserved for unstructured reflections only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
class CognitionMatrixEntry:
    player_id: str
    role_probabilities: dict[str, float] = field(default_factory=dict)
    faction_read: str = "unknown"
    trust: float = 0.5
    key_evidence: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "role_probabilities": dict(self.role_probabilities),
            "faction_read": self.faction_read,
            "trust": self.trust,
            "key_evidence": list(self.key_evidence),
            "open_questions": list(self.open_questions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitionMatrixEntry":
        return cls(
            player_id=data["player_id"],
            role_probabilities=data.get("role_probabilities", {}),
            faction_read=data.get("faction_read", "unknown"),
            trust=data.get("trust", 0.5),
            key_evidence=data.get("key_evidence", []),
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
        for attr in ("logic", "deception", "leadership", "credibility",
                      "learning_rate", "risk_preference"):
            if attr in deltas:
                new_val = getattr(self, attr) + deltas[attr]
                setattr(self, attr, max(0.0, min(1.0, new_val)))

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
# Reflection — long-term unstructured experience
# ---------------------------------------------------------------------------

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
    tags: list[str] = field(default_factory=list)
    situation: str = ""
    max_results: int = 5
    faction_won: bool | None = None
