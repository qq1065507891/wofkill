"""Contradiction Engine: detect stance reversals and vote conflicts.

Scans structured facts for logical contradictions in public behavior.
Provides attack vectors and defense alerts — never directly generates speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact


# ---------------------------------------------------------------------------
# Contradiction data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContradictionAlert:
    """A detected contradiction in a player's public behavior."""
    player_id: str
    alert_type: str  # "stance_reversal", "vote_conflict", "claim_conflict", "badge_flow_conflict"
    priority: str  # "high", "medium", "low"
    description: str
    evidence: tuple[dict[str, Any], ...] = ()
    day_range: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Contradiction Engine
# ---------------------------------------------------------------------------

class ContradictionEngine:
    """Detects contradictions from structured facts.

    Contradictions are potential attack vectors or defense alerts.
    They represent behavioral inconsistencies, not rule violations.
    """

    def __init__(self) -> None:
        pass

    def detect(
        self,
        facts: list[StructuredFact],
        current_day: int,
    ) -> list[ContradictionAlert]:
        """Run all contradiction detectors on visible facts."""
        alerts: list[ContradictionAlert] = []

        alerts.extend(self._detect_stance_reversals(facts, current_day))
        alerts.extend(self._detect_vote_conflicts(facts, current_day))
        alerts.extend(self._detect_claim_conflicts(facts, current_day))
        alerts.extend(self._detect_claim_contradictions(facts, current_day))

        return alerts

    def _detect_stance_reversals(
        self,
        facts: list[StructuredFact],
        current_day: int,
    ) -> list[ContradictionAlert]:
        """Detect when a player defends then attacks the same target."""
        alerts: list[ContradictionAlert] = []
        # Collect speech facts by player
        speeches_by_player: dict[str, list[StructuredFact]] = {}
        for f in facts:
            if f.fact_type == "speech" and f.source_player:
                speeches_by_player.setdefault(f.source_player, []).append(f)

        # Collect claims
        claims_by_player: dict[str, list[StructuredFact]] = {}
        for f in facts:
            if f.fact_type.startswith("claimed_") and f.source_player:
                claims_by_player.setdefault(f.source_player, []).append(f)

        # Check: player claims X about target on day N, opposite on day N+1
        for pid, claims in claims_by_player.items():
            target_claims: dict[str, list[StructuredFact]] = {}
            for c in claims:
                tgt = c.target_player
                if tgt:
                    target_claims.setdefault(tgt, []).append(c)

            for tgt, t_claims in target_claims.items():
                if len(t_claims) < 2:
                    continue
                # Look for opposing values (good vs wolf, defend vs attack)
                values = [c.value.lower() for c in t_claims]
                days = [c.day for c in t_claims]
                has_good = any(v in ("good", "good_lean") for v in values)
                has_wolf = any(v in ("wolf", "wolf_lean", "werewolf") for v in values)
                if has_good and has_wolf and len(set(days)) > 1:
                    alerts.append(ContradictionAlert(
                        player_id=pid,
                        alert_type="stance_reversal",
                        priority="high",
                        description=(
                            f"{pid}: stance reversal on {tgt} — "
                            f"claimed both good and wolf lean"
                        ),
                        evidence=tuple(
                            {"day": c.day, "value": c.value, "target": c.target_player}
                            for c in t_claims
                        ),
                        day_range=(min(days), max(days)),
                    ))

        return alerts

    def _detect_vote_conflicts(
        self,
        facts: list[StructuredFact],
        current_day: int,
    ) -> list[ContradictionAlert]:
        """Detect vote vs claim contradictions."""
        alerts: list[ContradictionAlert] = []

        # Collect votes and claims
        votes: dict[str, list[StructuredFact]] = {}  # voter → votes
        suspect_claims: dict[str, list[StructuredFact]] = {}  # claimer → suspect claims

        for f in facts:
            if f.fact_type == "vote" and f.source_player:
                votes.setdefault(f.source_player, []).append(f)
            if f.fact_type == "claimed_suspect" and f.source_player:
                suspect_claims.setdefault(f.source_player, []).append(f)

        # Check: player claims X is wolf but votes for Y instead (same day only)
        for pid, claims in suspect_claims.items():
            claim_days: dict[int, set[str]] = {}
            for c in claims:
                if c.target_player:
                    claim_days.setdefault(c.day, set()).add(c.target_player)
            player_votes = votes.get(pid, [])
            vote_days: dict[int, set[str]] = {}
            for v in player_votes:
                if v.target_player:
                    vote_days.setdefault(v.day, set()).add(v.target_player)

            for day, claimed_targets in claim_days.items():
                voted_targets = vote_days.get(day, set())
                if claimed_targets and voted_targets and not (claimed_targets & voted_targets):
                    alerts.append(ContradictionAlert(
                        player_id=pid,
                        alert_type="vote_conflict",
                        priority="medium",
                        description=(
                            f"{pid}: claimed {claimed_targets} as suspect "
                            f"but voted for {voted_targets} (day {day})"
                        ),
                        evidence=tuple(
                            {"day": f.day, "target": f.target_player, "type": f.fact_type}
                            for f in claims + player_votes
                            if f.day == day
                        ),
                    ))

        return alerts

    def _detect_claim_conflicts(
        self,
        facts: list[StructuredFact],
        current_day: int,
    ) -> list[ContradictionAlert]:
        """Detect conflicting role claims (e.g., two players claim seer)."""
        alerts: list[ContradictionAlert] = []
        role_claims: dict[str, list[str]] = {}  # role → list of claimers

        for f in facts:
            if f.fact_type == "claimed_role" and f.source_player:
                role = f.value.lower()
                role_claims.setdefault(role, []).append(f.source_player)

        for role, claimers in role_claims.items():
            if len(set(claimers)) > 1:
                alerts.append(ContradictionAlert(
                    player_id=",".join(set(claimers)),
                    alert_type="claim_conflict",
                    priority="high",
                    description=(
                        f"Multiple players claimed {role}: "
                        f"{', '.join(set(claimers))}"
                    ),
                    evidence=(
                        {"role": role, "claimers": list(set(claimers))},
                    ),
                ))

        return alerts

    def _detect_claim_contradictions(
        self,
        facts: list[StructuredFact],
        current_day: int,
    ) -> list[ContradictionAlert]:
        """Detect players who contradict their own role claims.

        For example, a player who claimed seer but later says "等预言家跳出来"
        is contradicting their own claim.
        """
        alerts: list[ContradictionAlert] = []

        # Build role claim map: player -> [(day, claimed_role)]
        role_claims: dict[str, list[tuple[int, str]]] = {}
        for f in facts:
            if f.fact_type == "claimed_role" and f.source_player:
                role_claims.setdefault(f.source_player, []).append((f.day, f.value))

        # Detect seer claimant saying things that contradict being seer
        contradiction_phrases = ("等预言家", "预言家在哪里", "等真预言家")
        for player_id, claims in role_claims.items():
            is_seer = any(role == "seer" for _, role in claims)
            if not is_seer:
                continue

            claim_day = next((d for d, r in claims if r == "seer"), 0)

            for f in facts:
                if f.source_player != player_id:
                    continue
                # Check speech text for contradiction phrases
                text = ""
                if f.fact_type == "speech":
                    text = f.value
                if not text:
                    continue

                for phrase in contradiction_phrases:
                    if phrase in text and f.day > claim_day:
                        alerts.append(ContradictionAlert(
                            player_id=player_id,
                            alert_type="claim_contradiction",
                            priority="high",
                            description=(
                                f"{player_id} claimed seer on day {claim_day} "
                                f"but says '{text}' on day {f.day}"
                            ),
                            evidence=(
                                {"day": claim_day, "event": "claimed seer"},
                                {"day": f.day, "event": text},
                            ),
                            day_range=(claim_day, f.day),
                        ))
                        break  # One contradiction per fact is enough

        return alerts
