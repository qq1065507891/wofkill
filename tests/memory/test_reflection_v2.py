from __future__ import annotations

import pytest

from werewolf_agent.memory.reflection import (
    ReflectionMemory,
    ReflectionQualityGate,
    ReflectionSynthesizer,
)
from werewolf_agent.memory.schemas import (
    CrossGameQuery,
    ReflectionEntryV2,
    ReflectionQualityStatus,
    ReviewReport,
)


def _v2_entry(**overrides):
    data = {
        "schema_version": 2,
        "entry_id": "reflection_g1_p01",
        "game_id": "g1",
        "player_id": "p01",
        "role": "seer",
        "faction": "good",
        "faction_won": False,
        "quality_score": 0.0,
        "quality_status": "review_only",
        "quality_flags": [],
        "situation_signature": {
            "role": "seer",
            "faction": "good",
            "outcome": "loss",
            "phase_focus": ["sheriff_speech", "vote"],
            "game_patterns": ["seer_counterclaim"],
        },
        "mistake_patterns": [
            {
                "category": "vote_mistake",
                "trigger": "双预言家对跳且警徽流解释不完整",
                "wrong_action": "只看发言气势就站边",
                "better_action": "先比较验人时间线、警徽流和票型承接",
                "fact_basis": "auto_review",
                "auto_verified": True,
                "corrected_from_llm": False,
            }
        ],
        "preserved_strengths": [
            {
                "category": "speech_quality",
                "behavior": "能明确说明验人心路",
                "reuse_condition": "警上起跳或被质疑时",
            }
        ],
        "actionable_advice": [
            "投票前先列双方验人时间线、警徽流和票型承接。"
        ],
        "avoid_next_time": ["不要因为发言强势就默认预言家可信。"],
        "prompt_card": {
            "theme": "对跳局先核验警徽流",
            "lesson": (
                "你过去在对跳局中过早相信气势强的一方。下次先核验验人时间线、"
                "警徽流和票型承接。"
            ),
            "trigger_signals": ["双预言家对跳", "警徽流解释不完整"],
            "recommended_action": "发言或投票前先列对比表，再给站边结论。",
            "misuse_risk": "不要把历史对跳经验直接映射到本局玩家身份。",
            "fact_basis": "auto_review",
            "auto_verified": True,
        },
        "source": {
            "llm_self_review": "我被强势发言带偏，但能复盘警徽流。",
            "auto_review_summary": "角色=seer，阵营失败，错误=1项",
            "merged_by": "reflection_synthesizer_v2",
        },
    }
    data.update(overrides)
    return ReflectionEntryV2.model_validate(data)


def test_v2_reflection_accepts_complete_schema() -> None:
    entry = _v2_entry()

    assert entry.schema_version == 2
    assert entry.prompt_card.theme == "对跳局先核验警徽流"
    assert entry.mistake_patterns[0].fact_basis == "auto_review"


def test_v2_reflection_rejects_missing_prompt_card() -> None:
    data = _v2_entry().model_dump()
    data.pop("prompt_card")

    with pytest.raises(Exception):
        ReflectionEntryV2.model_validate(data)


def test_quality_gate_approves_specific_actionable_reflection() -> None:
    gate = ReflectionQualityGate()

    gated = gate.evaluate(_v2_entry())

    assert gated.quality_status == ReflectionQualityStatus.APPROVED
    assert gated.quality_score >= 0.70
    assert gated.quality_flags == []


def test_quality_gate_rejects_player_id_leak_in_prompt_visible_fields() -> None:
    gate = ReflectionQualityGate()
    entry = _v2_entry(
        prompt_card={
            **_v2_entry().prompt_card.model_dump(),
            "lesson": "上次 p03 悍跳时你被带偏。",
        }
    )

    gated = gate.evaluate(entry)

    assert gated.quality_status == ReflectionQualityStatus.REJECTED
    assert "player_id_leak" in gated.quality_flags


def test_quality_gate_rejects_cjk_adjacent_player_id_leak() -> None:
    gate = ReflectionQualityGate()
    entry = _v2_entry(
        prompt_card={
            **_v2_entry().prompt_card.model_dump(),
            "lesson": "上次p03悍跳时你被带偏。",
        }
    )

    gated = gate.evaluate(entry)

    assert gated.quality_status == ReflectionQualityStatus.REJECTED
    assert "player_id_leak" in gated.quality_flags


def test_quality_gate_marks_generic_text_review_only_or_rejected() -> None:
    gate = ReflectionQualityGate()
    entry = _v2_entry(
        mistake_patterns=[],
        preserved_strengths=[],
        actionable_advice=["复盘失败对局，关注关键转折点的信息缺失"],
        prompt_card={
            "theme": "复盘失败",
            "lesson": "复盘失败对局，关注关键转折点的信息缺失。",
            "trigger_signals": [],
            "recommended_action": "关注关键转折点的信息缺失。",
            "misuse_risk": "不要直接套用。",
            "fact_basis": "llm_transferable",
            "auto_verified": False,
        },
    )

    gated = gate.evaluate(entry)

    assert gated.quality_status in {
        ReflectionQualityStatus.REVIEW_ONLY,
        ReflectionQualityStatus.REJECTED,
    }
    assert gated.quality_score < 0.70


def test_duplicate_reflection_is_penalized() -> None:
    gate = ReflectionQualityGate(existing_entries=[_v2_entry(quality_status="approved", quality_score=0.85)])

    duplicate = gate.evaluate(_v2_entry(entry_id="reflection_g2_p01", game_id="g2"))

    assert "duplicate" in duplicate.quality_flags
    assert duplicate.quality_score < 0.70
    assert duplicate.quality_status == ReflectionQualityStatus.REVIEW_ONLY


def test_reflection_memory_live_query_returns_only_approved_v2() -> None:
    memory = ReflectionMemory()
    approved = _v2_entry(quality_status="approved", quality_score=0.85)
    review_only = _v2_entry(
        entry_id="reflection_g2_p01",
        game_id="g2",
        quality_status="review_only",
        quality_score=0.55,
    )
    rejected = _v2_entry(
        entry_id="reflection_g3_p01",
        game_id="g3",
        quality_status="rejected",
        quality_score=0.2,
    )
    memory.store_v2(approved)
    memory.store_v2(review_only)
    memory.store_v2(rejected)

    results = memory.query_live(CrossGameQuery(player_id="p01", role="seer"))

    assert [r.entry_id for r in results] == [approved.entry_id]


def test_synthesizer_produces_one_v2_entry_with_auto_review_precedence() -> None:
    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="seer",
        faction_won=False,
        error_analysis=["误判 p03 为 villager（实际 werewolf）"],
        successful_strategies=["发言中能说明验人心路"],
        improvement_suggestions=["投票前先核验警徽流和票型承接"],
        summary="角色=seer，阵营失败，错误=1项",
    )
    synthesizer = ReflectionSynthesizer()

    entry = synthesizer.synthesize(
        llm_self_review="我觉得 p03 是好人，所以我没有犯错。",
        review_report=report,
        faction="good",
    )

    assert entry.schema_version == 2
    assert entry.entry_id == "reflection_g1_p01"
    assert entry.source.merged_by == "reflection_synthesizer_v2"
    assert entry.mistake_patterns
    assert all("p03" not in item for item in entry.prompt_visible_texts())
    assert any(p.corrected_from_llm for p in entry.mistake_patterns)


def test_prompt_visible_texts_excludes_wrong_action_with_truth_token() -> None:
    # wrong_action carries deterministic ground-truth ("实际预言家"); it must
    # NOT be in prompt_visible_texts, so the quality gate's truth-claim scan
    # is not bypassed by auto_verified.
    entry = _v2_entry(
        mistake_patterns=[
            {
                "category": "vote_mistake",
                "trigger": "双预言家对跳",
                "wrong_action": "误判 某玩家 为 狼人（实际 预言家）",
                "better_action": "先核验警徽流",
                "fact_basis": "auto_review",
                "auto_verified": True,
                "corrected_from_llm": False,
            }
        ],
    )
    visible = entry.prompt_visible_texts()
    joined = "\n".join(visible)
    # The truth-bearing wrong_action text must not appear in the prompt-visible set.
    assert "实际 预言家" not in joined
    assert "误判 某玩家 为 狼人" not in joined


def test_synthesize_structures_llm_preserved_strengths() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="seer",
        faction_won=True,
    )
    llm_review = (
        "【投票错误】我 D2 站错边。\n"
        "【保留的优点】本局做对的:\n"
        "- N2 用解药救了警长,后续归票翻盘\n"
        "- D3 提前质疑悍跳狼警徽流时间线,被采信\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )
    behaviors = [s.behavior for s in entry.preserved_strengths]
    # The LLM-articulated strengths are now structured (not just stashed in source).
    assert any("解药救了警长" in b for b in behaviors)
    assert any("警徽流" in b for b in behaviors)


def test_synthesize_llm_strength_drops_truth_tokens() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="villager", faction_won=True)
    llm_review = (
        "【保留的优点】\n"
        "- 某玩家实际是预言家我保住了他\n"
        "- 我坚持证据优先的站边\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )
    behaviors = [s.behavior for s in entry.preserved_strengths]
    assert not any("实际" in b for b in behaviors)
    assert any("证据优先" in b for b in behaviors)


def test_synthesize_keeps_deterministic_strengths_when_no_llm_section() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="seer", faction_won=True)
    report.successful_strategies.append("角色判断准确率高，继续保持基于证据的推理方式")
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review="没有段落头的纯文本反思。",
        review_report=report,
        faction="good",
    )
    behaviors = [s.behavior for s in entry.preserved_strengths]
    assert any("证据" in b for b in behaviors)


def test_synthesize_marks_corrected_for_varied_denial_phrasings() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="villager", faction_won=False)
    report.error_analysis.append("误判 某玩家 为 狼人（实际 预言家），最佳角色概率 0.80")
    for denial in (
        "我这局没什么问题",
        "我的判断都挺准的",
        "没有明显失误",
        "都还好",
    ):
        entry = ReflectionSynthesizer().synthesize(
            llm_self_review=denial,
            review_report=report,
            faction="good",
        )
        assert entry.mistake_patterns, f"expected patterns for denial={denial!r}"
        assert entry.mistake_patterns[0].corrected_from_llm is True, (
            f"expected corrected_from_llm=True for denial={denial!r}"
        )


def test_synthesize_does_not_mark_corrected_for_error_confessions() -> None:
    # "我的判断都错了" is a CONFESSION, not a denial — must NOT set corrected_from_llm.
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="villager", faction_won=False)
    report.error_analysis.append("误判 某玩家 为 狼人（实际 预言家），最佳角色概率 0.80")
    for confession in ("我的判断都错了", "我的判断都不准", "我的判断都偏了"):
        entry = ReflectionSynthesizer().synthesize(
            llm_self_review=confession,
            review_report=report,
            faction="good",
        )
        assert entry.mistake_patterns, f"expected patterns for confession={confession!r}"
        assert entry.mistake_patterns[0].corrected_from_llm is False, (
            f"confession {confession!r} must not be treated as a denial"
        )
