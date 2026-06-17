"""Seer claim credibility: deterministic credibility layer for seer lines.

Sits between visible structured facts and belief updates. Scores seer
claimants from public evidence only (no ground truth), in fact order so a
vote is judged only against claims that came before it.

Spec: docs/superpowers/specs/2026-06-17-seer-claim-credibility-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.public_evidence import is_good_result, is_wolf_result
from werewolf_agent.cognition.world_state import StructuredFact


_BASE_SCORE = 0.50
_DELTAS = {
    "first_seer_claim": 0.05,
    "has_check": 0.08,
    "no_dup_target": 0.05,
    "vote_follows_black": 0.10,
    "no_attack_gold": 0.05,
    "badge_flow": 0.05,
    "public_result_supports": 0.15,
    "multi_claimants": -0.15,
    "vote_against_black": -0.15,
    "attack_gold": -0.15,
    "changed_role": -0.20,
    "check_contradicts_public": -0.25,
    "stance_reversal": -0.10,
    "missing_vote_support": -0.05,
}
_SEVERE_PENALTIES = ("check_contradicts_public", "changed_role")


@dataclass
class ClaimCredibility:
    claimant: str
    claimed_role: str
    score: float
    confidence: float
    status: str
    evidence: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)


@dataclass
class SeerCheckClaim:
    claimant: str
    target: str
    result: str
    day: int
    evidence_ref: str = ""


@dataclass
class SeerLine:
    claimant: str
    checks: list[SeerCheckClaim] = field(default_factory=list)
    badge_flow: list[str] = field(default_factory=list)
    credibility: ClaimCredibility = field(
        default_factory=lambda: ClaimCredibility("", "", _BASE_SCORE, 0.25, "weak")
    )


class SeerClaimCredibilityEngine:
    """Deterministic seer-line credibility from visible public facts."""

    def __init__(self) -> None:
        self._lines: dict[str, SeerLine] = {}
        self._claimant_order: list[str] = []
        self._role_history: dict[str, list[str]] = {}
        self._gold_claims: dict[str, set[str]] = {}
        self._votes: list[StructuredFact] = []

    def _get_or_create_line(self, claimant: str) -> SeerLine:
        if claimant not in self._lines:
            self._lines[claimant] = SeerLine(claimant=claimant)
            self._claimant_order.append(claimant)
        return self._lines[claimant]

    def observe(self, fact: StructuredFact) -> None:
        ft = fact.fact_type
        src = fact.source_player
        if not src:
            return
        if ft == "claimed_role":
            role = (fact.value or "").lower()
            self._role_history.setdefault(src, []).append(role)
            if role == "seer":
                self._get_or_create_line(src)
        elif ft == "seer_check_claim":
            line = self._get_or_create_line(src)
            line.checks.append(SeerCheckClaim(
                claimant=src, target=fact.target_player or "",
                result=fact.value or "", day=fact.day,
            ))
        elif ft == "badge_flow_claim":
            line = self._get_or_create_line(src)
            order = fact.metadata.get("badge_flow_order") if fact.metadata else None
            if isinstance(order, list):
                line.badge_flow.extend(str(x) for x in order)
        elif ft == "claimed_good":
            if fact.target_player:
                self._gold_claims.setdefault(src, set()).add(fact.target_player)
        elif ft == "vote":
            self._votes.append(fact)

    def _raw_score(
        self, claimant: str
    ) -> tuple[float, float, list[str], list[str]]:
        line = self._lines[claimant]
        score = _BASE_SCORE
        evidence: list[str] = []
        penalties: list[str] = []

        if self._claimant_order and self._claimant_order[0] == claimant:
            score += _DELTAS["first_seer_claim"]
            evidence.append("first_seer_claim")
        if len(set(self._role_history.get(claimant, []))) > 1:
            score += _DELTAS["changed_role"]
            penalties.append("changed_role")
        if line.checks:
            score += _DELTAS["has_check"]
            evidence.append("has_check")
            targets = [c.target for c in line.checks]
            if len(set(targets)) == len(targets):
                score += _DELTAS["no_dup_target"]
                evidence.append("no_dup_target")
        if line.badge_flow:
            score += _DELTAS["badge_flow"]
            evidence.append("badge_flow")

        own_blacks = {c.target for c in line.checks if is_wolf_result(c.result)}
        own_golds = {
            c.target for c in line.checks if is_good_result(c.result)
        } | self._gold_claims.get(claimant, set())
        claimant_votes = [v for v in self._votes if v.source_player == claimant]
        voted = {v.target_player for v in claimant_votes if v.target_player}
        if own_blacks and voted:
            if voted & own_blacks:
                score += _DELTAS["vote_follows_black"]
                evidence.append("vote_follows_black")
            else:
                score += _DELTAS["vote_against_black"]
                penalties.append("vote_against_black")
        if own_golds and voted:
            if voted & own_golds:
                score += _DELTAS["attack_gold"]
                penalties.append("attack_gold")
            else:
                score += _DELTAS["no_attack_gold"]
                evidence.append("no_attack_gold")

        if len(self._lines) > 1:
            score += _DELTAS["multi_claimants"]
            penalties.append("multi_claimants")

        score = max(0.0, min(1.0, score))
        confidence = min(1.0, 0.25 + 0.15 * len(evidence))
        return score, confidence, evidence, penalties

    def _score_all(self) -> dict[str, ClaimCredibility]:
        raw = {c: self._raw_score(c) for c in self._lines}
        any_supported = any(
            s >= 0.65 and conf >= 0.55 for s, conf, _, _ in raw.values()
        )
        n = len(self._lines)
        result: dict[str, ClaimCredibility] = {}
        for claimant, (s, conf, ev, pen) in raw.items():
            if any(p in _SEVERE_PENALTIES for p in pen):
                status = "broken"
            elif n > 1 and not any_supported:
                status = "contested"
            elif s >= 0.65 and conf >= 0.55:
                status = "supported"
            elif s < 0.45:
                status = "weak"
            else:
                status = "uncontested"
            result[claimant] = ClaimCredibility(
                claimant=claimant, claimed_role="seer",
                score=round(s, 4), confidence=round(conf, 4),
                status=status, evidence=ev, penalties=pen,
            )
        return result

    def score_for(self, claimant: str) -> ClaimCredibility:
        if claimant not in self._lines:
            return ClaimCredibility(
                claimant=claimant, claimed_role="", score=0.0,
                confidence=0.0, status="weak",
            )
        return self._score_all()[claimant]

    def line_for(self, claimant: str) -> SeerLine | None:
        line = self._lines.get(claimant)
        if line is None:
            return None
        line.credibility = self.score_for(claimant)
        return line

    def prompt_summary(self, limit: int = 3) -> dict[str, Any]:
        scored = self._score_all()
        if not scored:
            return {}
        items = sorted(scored.items(), key=lambda kv: -kv[1].score)[:limit]
        return {
            "seer_lines": [
                {
                    "claimant": c,
                    "status": cred.status,
                    "score": cred.score,
                    "checks": [
                        f"{chk.result}:{chk.target}"
                        for chk in self._lines[c].checks[:3]
                    ],
                    "evidence": list(cred.evidence[:3]),
                    "penalties": list(cred.penalties[:3]),
                }
                for c, cred in items
            ]
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "lines": {
                c: {
                    "checks": [
                        {"target": chk.target, "result": chk.result, "day": chk.day}
                        for chk in line.checks
                    ],
                    "badge_flow": list(line.badge_flow),
                }
                for c, line in self._lines.items()
            },
            "claimant_order": list(self._claimant_order),
            "role_history": {c: list(r) for c, r in self._role_history.items()},
            "gold_claims": {c: sorted(t) for c, t in self._gold_claims.items()},
            "votes": [
                {"voter": v.source_player, "target": v.target_player, "day": v.day}
                for v in self._votes
            ],
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "SeerClaimCredibilityEngine":
        eng = cls()
        for c in snap.get("claimant_order", []):
            eng._get_or_create_line(c)
        for c, data in snap.get("lines", {}).items():
            line = eng._get_or_create_line(c)
            for chk in data.get("checks", []):
                line.checks.append(SeerCheckClaim(
                    claimant=c, target=chk.get("target", ""),
                    result=chk.get("result", ""), day=chk.get("day", 0),
                ))
            line.badge_flow.extend(data.get("badge_flow", []))
        eng._role_history = {
            c: list(r) for c, r in snap.get("role_history", {}).items()
        }
        eng._gold_claims = {
            c: set(t) for c, t in snap.get("gold_claims", {}).items()
        }
        eng._votes = [
            StructuredFact(
                fact_type="vote", source_player=v.get("voter", ""),
                target_player=v.get("target", ""), day=v.get("day", 0),
            )
            for v in snap.get("votes", [])
        ]
        return eng
