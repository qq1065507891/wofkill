"""Tests for public speech quality validation."""

import pytest
from werewolf_agent.runtime.speech_quality import (
    extract_speech_quality,
    validate_public_speech,
    build_speech_retry_hint,
    fallback_speech_with_basis,
)


class TestFillerSpeechRejection:
    """Filler speech without substance is rejected."""

    def test_rejects_filler_day_speech(self):
        """Speech like '再观察一下' fails validation."""
        result = validate_public_speech("再观察一下，先听后面", phase="day_discussion", context={})
        assert result["valid"] is False

    def test_rejects_empty_speech(self):
        result = validate_public_speech("", phase="day_discussion", context={})
        assert result["valid"] is False

    def test_rejects_observation_only(self):
        result = validate_public_speech("信息太少，我没什么可说的", phase="day_discussion", context={})
        assert result["valid"] is False

    def test_rejects_pure_deflection(self):
        result = validate_public_speech("我是个好人，大家相信我就行了", phase="day_discussion", context={})
        assert result["valid"] is False


class TestRequiredSpeechComponents:
    """Valid day speech must include stance, targets, vote leaning, evidence."""

    def test_day_speech_requires_stance_targets_and_evidence(self):
        """A complete speech with all required components passes."""
        speech = (
            "我是好人阵营。我怀疑p03，p03的发言前后矛盾，"
            "特别是昨天说保p05今天又投票给p05。我信任p07，"
            "p07的逻辑链完整。我倾向于投票p03。"
        )
        result = validate_public_speech(speech, phase="day_discussion", context={})
        assert result["valid"] is True

    def test_speech_with_stance_and_suspicion_passes(self):
        speech = "我是好人，我觉得p05是狼人，p05的警徽流不合理，我倾向投p05。保护p01。"
        result = validate_public_speech(speech, phase="day_discussion", context={})
        assert result["valid"] is True

    def test_speech_without_vote_leaning_fails(self):
        """Speech must include vote leaning."""
        speech = "我怀疑p03是狼人，p03发言有矛盾。"
        result = validate_public_speech(speech, phase="day_discussion", context={})
        # Should fail if no vote leaning
        assert "vote_leaning" in result.get("missing_fields", [])

    def test_speech_without_evidence_fails(self):
        speech = "我怀疑p03，投p03。"
        result = validate_public_speech(speech, phase="day_discussion", context={})
        # Very short speech without evidence basis
        assert result["valid"] is False or "evidence" in str(result.get("missing_fields", []))


class TestHighPressureSpeech:
    """Sheriff, PK, seer speeches have stronger requirements."""

    def test_high_pressure_speech_requires_claim_logic(self):
        """Sheriff/PK/seer speeches must include claim-related logic."""
        speech = "我是预言家，昨晚验p03是查杀，警徽流p05 p07。p03必须出局。"
        result = validate_public_speech(speech, phase="sheriff_speech", context={"is_claiming_role": True})
        assert result["valid"] is True

    def test_sheriff_speech_without_claim_fails(self):
        """Sheriff speech without role claim or counterclaim is weak."""
        speech = "我觉得大家都很好"
        result = validate_public_speech(speech, phase="sheriff_speech", context={})
        assert result["valid"] is False


class TestSpeechQualityExtraction:
    """extract_speech_quality identifies speech components."""

    def test_extracts_suspicion_targets(self):
        quality = extract_speech_quality("我怀疑p03和p05", phase="day_discussion")
        assert len(quality.get("suspicion_targets", [])) >= 1

    def test_extracts_protection_targets(self):
        quality = extract_speech_quality("我信任p07，保p01", phase="day_discussion")
        assert len(quality.get("protection_targets", [])) >= 1

    def test_extracts_vote_leaning(self):
        quality = extract_speech_quality("我倾向投p03", phase="day_discussion")
        assert quality.get("vote_leaning") is not None

    def test_extracts_evidence_basis(self):
        quality = extract_speech_quality("p03发言前后矛盾", phase="day_discussion")
        assert len(quality.get("evidence_bases", [])) >= 1


class TestRetryHint:
    """build_speech_retry_hint provides actionable guidance."""

    def test_hint_for_missing_vote_leaning(self):
        hint = build_speech_retry_hint(["vote_leaning"])
        assert "投票倾向" in hint or "vote_leaning" in hint

    def test_hint_for_multiple_missing(self):
        hint = build_speech_retry_hint(["vote_leaning", "evidence"])
        assert len(hint) > 10


class TestFallbackSpeech:
    """fallback_speech_with_basis creates a minimum viable speech."""

    def test_fallback_has_targets_and_evidence(self):
        context = {
            "own_id": "p01",
            "suspicion_candidates": ["p03", "p05"],
            "day": 2,
        }
        speech = fallback_speech_with_basis(context)
        assert "p03" in speech or "p05" in speech
        assert len(speech) > 20
