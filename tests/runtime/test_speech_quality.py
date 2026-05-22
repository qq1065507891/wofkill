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


class TestPeaceNightWitchReasoning:
    """Peace night reasoning must not misread public no-death as no wolf kill."""

    def test_rejects_peace_night_means_witch_cannot_know_kill_target(self):
        speech = (
            "作为村民，我要问p05：昨晚平安夜，你到底救了谁？"
            "如果你说不出具体是谁，你的女巫身份就是假的。"
            "如果p11是真的预言家，他应该质疑这个矛盾，而不是急着给p05发金水。"
            "我倾向投p05，因为平安夜根本没有人死，女巫不可能知道狼人刀了人。"
        )

        result = validate_public_speech(speech, phase="day_discussion", context={})

        assert result["valid"] is False
        assert "peace_night_witch_reasoning" in result["missing_fields"]
        assert "平安夜不等于无人被刀" in result["hint"]

    def test_accepts_reasonable_pressure_on_witch_claim(self):
        speech = (
            "我是好人阵营。平安夜可能是狼人空刀，也可能是女巫救人。"
            "我怀疑p05不是因为平安夜本身，而是p05前后发言矛盾：先说用了药，"
            "又不解释为什么暂时不公开银水。p05可以藏救人对象，但必须说明藏信息的收益。"
            "我倾向投p05，除非p05能解释用药逻辑。"
        )

        result = validate_public_speech(speech, phase="day_discussion", context={})

        assert result["valid"] is True


class TestPublicRecordGrounding:
    """Public-record claims must be backed by actual transcript text."""

    def test_rejects_unsupported_public_role_claim(self):
        speech = (
            "我是好人阵营。我怀疑p02，因为p02声称自己是狼人（Day 1公开记录）。"
            "这个发言矛盾很大，我倾向投p02。"
        )
        context = {
            "recent_transcript": [
                {"speaker": "p02", "text": "我是好人，今天先看票型。"},
            ],
            "public_summary": "D1 投票结果：p05 被放逐",
        }

        result = validate_public_speech(speech, phase="day_discussion", context=context)

        assert result["valid"] is False
        assert "public_record_grounding" in result["missing_fields"]
        assert "公开记录" in result["hint"]

    def test_accepts_supported_public_role_claim(self):
        speech = (
            "我是好人阵营。我怀疑p02，因为p02刚才说我是狼人，"
            "这和他的站边矛盾，我倾向投p02。"
        )
        context = {
            "recent_transcript": [
                {"speaker": "p02", "text": "我是狼人，这局我摊牌了。"},
            ],
        }

        result = validate_public_speech(speech, phase="day_discussion", context=context)

        assert result["valid"] is True


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
