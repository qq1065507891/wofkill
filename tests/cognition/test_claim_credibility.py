"""Seer claim credibility engine tests (spec 2026-06-17)."""

from __future__ import annotations

from werewolf_agent.cognition.claim_credibility import (
    SeerClaimCredibilityEngine,
)
from werewolf_agent.cognition.world_state import StructuredFact


def _seer_claim(pid="p08", day=1):
    return StructuredFact(fact_type="claimed_role", source_player=pid, value="seer", day=day)


def _check(claimant="p08", target="p01", result="wolf", day=1):
    return StructuredFact(fact_type="seer_check_claim", source_player=claimant,
                          target_player=target, value=result, day=day)


def _vote(voter="p08", target="p01", day=1):
    return StructuredFact(fact_type="vote", source_player=voter, target_player=target, day=day)


def test_single_seer_claimant_base_plus_first_bonus():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    c = eng.score_for("p08")
    assert c.claimed_role == "seer"
    assert round(c.score, 3) == 0.55  # 0.50 base + 0.05 first claim
    assert c.status == "uncontested"


def test_multiple_claimants_each_penalized_and_contested():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(_seer_claim("p01"))
    assert round(eng.score_for("p08").score, 3) == 0.40  # 0.50+0.05-0.15
    assert round(eng.score_for("p01").score, 3) == 0.35  # 0.50-0.15
    assert eng.score_for("p08").status == "contested"
    assert eng.score_for("p01").status == "contested"


def test_check_claim_bonus():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(_check("p08", "p01", "wolf"))
    # 0.50 + 0.05(first) + 0.08(has_check) + 0.05(no_dup) = 0.68
    assert round(eng.score_for("p08").score, 3) == 0.68


def test_vote_follows_own_black_check_supported():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(_check("p08", "p01", "wolf"))
    eng.observe(_vote("p08", "p01"))
    # 0.50+0.05+0.08+0.05+0.10 = 0.78
    assert round(eng.score_for("p08").score, 3) == 0.78
    assert eng.score_for("p08").status == "supported"


def test_vote_against_own_black_check_penalty():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(_check("p08", "p01", "wolf"))
    eng.observe(_vote("p08", "p02"))  # votes away from own black check
    # 0.50+0.05+0.08+0.05-0.15 = 0.53
    assert round(eng.score_for("p08").score, 3) == 0.53


def test_vote_against_own_gold_water_penalty():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(_check("p08", "p07", "good"))
    eng.observe(_vote("p08", "p07"))
    cred = eng.score_for("p08")
    assert "attack_gold" in cred.penalties
    assert cred.score < 0.68


def test_confidence_grows_with_evidence():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    c1 = eng.score_for("p08")
    eng.observe(_check("p08", "p01", "wolf"))
    c2 = eng.score_for("p08")
    assert c2.confidence > c1.confidence


def test_non_claimant_returns_weak_zero():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    c = eng.score_for("p09")
    assert c.score == 0.0
    assert c.status == "weak"


def test_badge_flow_bonus():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(StructuredFact(
        fact_type="badge_flow_claim", source_player="p08", target_player="p05",
        day=1, value="badge_flow", metadata={"badge_flow_order": ["p05", "p07"]},
    ))
    # 0.50 + 0.05(first) + 0.05(badge) = 0.60
    assert round(eng.score_for("p08").score, 3) == 0.60


def test_snapshot_round_trip():
    eng = SeerClaimCredibilityEngine()
    eng.observe(_seer_claim("p08"))
    eng.observe(_check("p08", "p01", "wolf"))
    eng.observe(_seer_claim("p01"))
    snap = eng.snapshot()
    eng2 = SeerClaimCredibilityEngine.from_snapshot(snap)
    assert round(eng2.score_for("p08").score, 3) == round(eng.score_for("p08").score, 3)
    assert eng2.score_for("p08").status == eng.score_for("p08").status
    assert eng2.score_for("p01").status == eng.score_for("p01").status
