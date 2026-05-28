"""Salience Engine: weight visible facts for prompt inclusion.

Weights are computed from recency, role relevance, phase relevance, and
contradiction potential. Facts are bucketed into:
- high: directly included in prompt
- medium: summary included in prompt
- low: kept in structured state / long-term memory only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact


# ---------------------------------------------------------------------------
# Salience data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SalientFact:
    """A fact annotated with its salience weight and bucket."""
    fact: StructuredFact
    weight: float
    bucket: str  # "high", "medium", "low"
    reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Weight computation helpers
# ---------------------------------------------------------------------------

# Fact types that are always high salience
_HIGH_PRIORITY_TYPES: set[str] = {
    "player_died",
    "player_exiled",
    "idiot_revealed",
    "self_destruct",
    "sheriff_elected",
    "badge_transferred",
    "badge_torn",
}

# Fact types about the current phase get a boost
_PHASE_RELEVANCE: dict[str, set[str]] = {
    "wolf_discussion_and_kill": {"wolf_kill_selected", "wolf_discussion"},
    "vote": {"vote", "claimed_suspect", "claimed_claim"},
    "speech": {"speech", "claimed_role", "claimed_suspect", "badge_flow_claim"},
    "seer_check": {"seer_check"},
    "witch_action": {"witch_antidote_used", "witch_poison_used"},
    "sheriff_registration": {"sheriff_registered", "sheriff_withdraw"},
    "sheriff_vote": {"sheriff_vote_tie", "sheriff_elected", "sheriff_no_election"},
}

# How many recent days count as "recent" (full weight)
_RECENCY_WINDOW_DAYS = 2


def _compute_weight(
    fact: StructuredFact,
    current_day: int,
    current_phase: str,
    viewer_role: str,
) -> tuple[float, list[str]]:
    """Compute raw salience weight [0..1] and reasons."""
    weight = 0.3  # baseline
    reasons: list[str] = []

    # High-priority fact types
    if fact.fact_type in _HIGH_PRIORITY_TYPES:
        weight += 0.4
        reasons.append(f"high_priority_type:{fact.fact_type}")

    # Recency boost: facts from recent days are more salient
    if fact.day > 0:
        days_ago = current_day - fact.day
        if days_ago <= 0:
            weight += 0.2
            reasons.append("current_day")
        elif days_ago <= _RECENCY_WINDOW_DAYS:
            weight += 0.1
            reasons.append("recent_day")

    # Phase relevance
    phase_types = _PHASE_RELEVANCE.get(current_phase, set())
    if fact.fact_type in phase_types:
        weight += 0.15
        reasons.append("phase_relevant")

    # Role-specific relevance
    if viewer_role == "werewolf" and fact.fact_type in {"wolf_kill_selected", "wolf_discussion"}:
        weight += 0.2
        reasons.append("role_relevant:wolf_team")
    if viewer_role == "seer" and fact.fact_type == "seer_check":
        weight += 0.2
        reasons.append("role_relevant:seer")
    if viewer_role == "witch" and fact.fact_type in {"witch_antidote_used", "witch_poison_used"}:
        weight += 0.2
        reasons.append("role_relevant:witch")

    # Claims about specific players get a boost
    if fact.fact_type.startswith("claimed_") and fact.target_player:
        weight += 0.1
        reasons.append("targeted_claim")

    # Night 0/early game facts lose weight over time
    if fact.day == 0 and fact.night <= 1 and current_day > 2:
        weight -= 0.1
        reasons.append("early_game_decay")

    return min(max(weight, 0.0), 1.0), reasons


def _bucket(weight: float, high: float = 0.7, medium: float = 0.4) -> str:
    if weight >= high:
        return "high"
    if weight >= medium:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Salience Engine
# ---------------------------------------------------------------------------

class SalienceEngine:
    """Weights visible facts and buckets them for prompt inclusion."""

    def __init__(
        self,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def weight_facts(
        self,
        facts: list[StructuredFact],
        current_day: int,
        current_phase: str,
        viewer_role: str,
    ) -> list[SalientFact]:
        """Weight all visible facts and return them sorted by weight descending."""
        salient: list[SalientFact] = []
        for fact in facts:
            weight, reasons = _compute_weight(
                fact, current_day, current_phase, viewer_role,
            )
            bucket = _bucket(weight, self.high_threshold, self.medium_threshold)
            salient.append(SalientFact(
                fact=fact,
                weight=weight,
                bucket=bucket,
                reasons=tuple(reasons),
            ))
        salient.sort(key=lambda s: s.weight, reverse=True)
        return salient

    def filter_by_bucket(
        self,
        weighted: list[SalientFact],
        max_bucket: str = "medium",
    ) -> list[SalientFact]:
        """Return facts at or above the given bucket level."""
        bucket_order = {"high": 3, "medium": 2, "low": 1}
        min_level = bucket_order.get(max_bucket, 2)
        return [
            s for s in weighted
            if bucket_order.get(s.bucket, 0) >= min_level
        ]
