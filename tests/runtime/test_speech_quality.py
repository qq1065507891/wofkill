# -*- coding: utf-8 -*-
"""
测试公开发言质量校验，包括模板发言、平安夜误推理和公开记录引用。

作者: Project contributors
修改日期: 2026-07-16
"""

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


class TestIntentAwareSpeechRequirements:
    def test_stand_with_seer_does_not_require_suspicion_or_vote(self):
        speech = "我明确站边p06，因为他的查验结果和警徽流能够互相印证。"

        result = validate_public_speech(
            speech,
            phase="day_discussion",
            context={"intent": "stand_with_seer", "target_id": "p06"},
        )

        assert result["valid"] is True
        assert "suspicion_target" not in result["missing_fields"]
        assert "vote_leaning" not in result["missing_fields"]

    def test_question_target_requires_target_and_evidence_not_vote(self):
        speech = "p12，你刚才说信息完整就可信，这个判断的具体依据是什么？"

        result = validate_public_speech(
            speech,
            phase="day_discussion",
            context={"intent": "question_target", "target_id": "p12"},
        )

        assert result["valid"] is True
        assert "vote_leaning" not in result["missing_fields"]

    def test_info_synthesis_accepts_multi_player_analysis_without_vote(self):
        speech = "p03站边p06，p04反对p06；两人的票型和警徽流需要一起核对。"

        result = validate_public_speech(
            speech,
            phase="day_discussion",
            context={"intent": "info_synthesis"},
        )

        assert result["valid"] is True

    def test_info_synthesis_accepts_chinese_name_list_as_multi_entity(self):
        speech = (
            "警上六人报名，这个结构值得关注。当前已知信息："
            "上警名单包含我、陈思远、赵猛、周烈、沈墨、陆无声、郑铭，共六人。"
            "我会观察谁在回应质疑时出现前后矛盾，谁的站边理由经不起推敲。"
        )

        result = validate_public_speech(
            speech,
            phase="sheriff_speech",
            context={"intent": "info_synthesis"},
        )

        assert result["valid"] is True
        assert "multi_entity" not in result["missing_fields"]

    def test_push_vote_still_requires_vote_leaning(self):
        speech = "p03的发言前后矛盾，我会继续关注。"

        result = validate_public_speech(
            speech,
            phase="day_discussion",
            context={"intent": "push_vote", "target_id": "p03"},
        )

        assert result["valid"] is False
        assert "vote_leaning" in result["missing_fields"]


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

    def test_claim_pattern_includes_all_roles(self):
        """D-12: a public claim to 村民 or 混血儿 must also count as
        a valid claim during a high-pressure phase.

        Pre-fix the regex only accepted 预言家|女巫|猎人|白痴, so a
        villager saying "我是村民" was misclassified as a missing
        claim, polluting the missing_fields list.
        """
        # Villager claim in sheriff / PK phase
        speech_v = "我是村民，我听了所有人的发言，我怀疑p03，倾向投p03"
        result_v = validate_public_speech(
            speech_v, phase="sheriff_speech", context={"is_claiming_role": True},
        )
        assert "claim_logic" not in result_v.get("missing_fields", []), (
            f"villager claim should not trigger claim_logic miss; "
            f"got missing: {result_v.get('missing_fields')}"
        )

        # Hybrid claim in PK phase
        speech_h = "我是混血儿，我主人是p05，我分析p03更像狼，倾向投p03"
        result_h = validate_public_speech(
            speech_h, phase="pk_speech", context={"is_claiming_role": True},
        )
        assert "claim_logic" not in result_h.get("missing_fields", []), (
            f"hybrid claim should not trigger claim_logic miss; "
            f"got missing: {result_h.get('missing_fields')}"
        )


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

    def test_rejects_peace_night_used_to_discredit_seer_check(self):
        speech = (
            "我是好人阵营。我怀疑p11，因为首夜平安夜没人死，"
            "他却说自己验出p03是狼人。如果他解释不清狼人为什么没人死，"
            "那他的验人结论就靠不住，我倾向投p11。"
        )

        result = validate_public_speech(speech, phase="day_discussion", context={})

        assert result["valid"] is False
        assert "peace_night_seer_reasoning" in result["missing_fields"]
        assert "不能用“平安夜没人死”否定预言家验人" in result["hint"]

    def test_accepts_seer_pressure_that_separates_peace_night_from_check(self):
        speech = (
            "我是好人阵营。平安夜只代表公开无人死亡，不影响预言家验人。"
            "我怀疑p11不是因为平安夜本身，而是他先报p03查杀后没有兑现警徽流，"
            "且没有解释为什么验p03。我倾向投p11。"
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

    def test_rejects_unsupported_public_death_claim(self):
        speech = (
            "我是好人阵营。我怀疑p12，因为p12说p02在死因报告里称p03被狼刀，"
            "但公开记录里没有这句话。这个逻辑矛盾很大，我倾向投p12。"
        )
        context = {
            "recent_transcript": [
                {"speaker": "p02", "text": "p03和p11对跳预言家，我先看警徽流。"},
                {"speaker": "p12", "text": "我认为p11需要补充警徽流。"},
            ],
            "public_summary": "昨夜是平安夜，无人死亡。",
        }

        result = validate_public_speech(speech, phase="day_discussion", context=context)

        assert result["valid"] is False
        assert "public_record_grounding" in result["missing_fields"]


    def test_accepts_supported_public_death_claim(self):
        speech = (
            "我是好人阵营。我怀疑p12，因为p02刚才说p03被狼刀，"
            "但公开死讯是平安夜，这个公开陈述需要p02解释。我倾向投p12。"
        )
        context = {
            "recent_transcript": [
                {"speaker": "p02", "text": "我认为p03被狼刀，但这个信息还需要核对。"},
            ],
            "public_summary": "昨夜是平安夜，无人死亡。",
        }

        result = validate_public_speech(speech, phase="day_discussion", context=context)

        assert result["valid"] is True

    def test_accepts_attributed_restatement_of_public_player_claim(self):
        """p11 转述 p05 的公开声明，不应被误判为系统确认事实。"""
        speech = (
            "我是好人阵营。我怀疑p08，因为p05刚才声称自己是女巫，"
            "而p08无视这条公开声明，逻辑矛盾，我倾向投p08。"
        )
        context = {
            "recent_transcript": [
                {"speaker": "p05", "text": "我是女巫，但暂时不公开药水。"},
            ],
        }
        result = validate_public_speech(speech, phase="day_discussion", context=context)
        assert result["valid"] is True

    def test_rejects_unsupported_system_fact_but_preserves_valid_attribution(self):
        speech = (
            "我是好人阵营。我怀疑p08，因为p05刚才声称自己是女巫，"
            "而且系统确认p08是狼人，这个逻辑矛盾，我倾向投p08。"
        )
        context = {
            "recent_transcript": [
                {"speaker": "p05", "text": "我是女巫，但暂时不公开药水。"},
            ],
        }
        result = validate_public_speech(speech, phase="day_discussion", context=context)
        assert result["valid"] is False
        assert "public_record_grounding" in result["missing_fields"]


@pytest.mark.parametrize(
    ("text", "unsupported"),
    [
        ("不能否认系统确认p08是狼人，我怀疑p08。", True),
        ("没有理由不信系统确认p08是狼人，我怀疑p08。", True),
        ("系统没有确认p08是狼人，我只是怀疑p08。", False),
        ("不能说系统确认p08是狼人，我只是怀疑p08。", False),
    ],
)
def test_speech_quality_reuses_authoritative_negation_scope(
    text: str, unsupported: bool
) -> None:
    from werewolf_agent.runtime.speech_quality import validate_public_speech

    result = validate_public_speech(
        text,
        context={"intent": "question_target", "target_id": "p08", "day_number": 1},
    )

    assert ("public_record_grounding" in result["missing_fields"]) is unsupported


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


def test_non_empty_terminal_fallback_does_not_count_as_model_success() -> None:
    from scripts.run_real_game import compute_game_quality_score
    from werewolf_agent.core.models import GameEvent, GameState, PlayerState
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    state = GameState(
        game_id="g-terminal-speech",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p01",
                "text": "[FALLBACK]普通发言仅基于公开信息：昨夜无人出局。",
            }),
            GameEvent(type="action_trace_audit", payload={
                "task_type": "speech",
                "action_trace": {
                    "generated_by": "terminal_fallback",
                    "decision_outcome": "terminal_fallback",
                    "terminal_failure_code": "schema_validation",
                    "original_failure_code": "schema_validation",
                    "failure_stage": "protocol",
                    "fallback_kind": "ordinary_speech",
                    "semantic_repair_audit": {"success": False},
                },
            }),
        ],
    )

    quality = compute_game_quality_score(project_acceptance_game(state))

    assert quality["speech_non_empty_rate"] == 1.0
    assert quality["speech_model_success_rate"] == 0.0
    assert quality["speech_terminal_fallback_rate"] == 1.0
    assert quality["speech_semantic_acceptance_rate"] == 0.0
