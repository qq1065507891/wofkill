"""Tests for seer claim validation (1-check-per-night rule)."""

from __future__ import annotations

from werewolf_agent.runtime.seer_claim_validator import (
    extract_seer_claims,
    validate_seer_claim,
)


class TestExtractSeerClaims:
    def test_single_claim_n1(self):
        claims = extract_seer_claims("我在第1夜查了p04是好人")
        assert len(claims) == 1
        assert claims[0]["night"] == 1
        assert claims[0]["target_id"] == "p04"

    def test_two_claims_same_night(self):
        claims = extract_seer_claims("我第1夜查了p04是村民，也查了p09是村民")
        assert len(claims) == 2
        assert all(c["night"] == 1 for c in claims)

    def test_empty_speech(self):
        assert extract_seer_claims("") == []

    def test_no_claim_in_speech(self):
        assert extract_seer_claims("我觉得p04的发言有矛盾") == []


class TestValidateSeerClaim:
    def test_single_claim_passes(self):
        assert validate_seer_claim("我查了p04是好人", day_number=1) is None

    def test_two_claims_same_night_fails(self):
        err = validate_seer_claim(
            "我第1夜查了p04是村民，也查了p09是村民", day_number=1
        )
        assert err is not None
        assert "违反" in err

    def test_future_night_claim_fails(self):
        err = validate_seer_claim("我第3夜查了p05是狼人", day_number=1)
        assert err is not None
        assert "未来" in err or "不可能" in err

    def test_night_zero_claim_fails(self):
        err = validate_seer_claim("我第0夜查了p05是狼人", day_number=1)
        assert err is not None

    def test_no_claim_passes(self):
        assert validate_seer_claim("我觉得p04的发言有矛盾", day_number=1) is None
