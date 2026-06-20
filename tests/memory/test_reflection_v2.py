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


# ---------------------------------------------------------------------------
# _extract_llm_mistakes — parse the 6 LLM mistake sections
# ---------------------------------------------------------------------------

_FULL_SIX_SECTIONS = """【投票错误】
- 没有等预言家报查验就投票
【信息缺失】
- 忽略了2号玩家警徽流打法的承接
【神职执行】
- 女巫第一晚就浪费了解药
【悍跳分析】
- 被对跳预言家的气势唬住没有核验时间线
【暴露原因】
- 发言时太早亮出神职身份导致被针对
【角色分工】
- 没有和狼队友对齐今晚的刀口目标
"""


def test_extract_llm_mistakes_parses_all_six_sections() -> None:
    patterns = ReflectionSynthesizer._extract_llm_mistakes(_FULL_SIX_SECTIONS, role="seer")
    # The 6 sections yield 4 distinct categories, but cap=3 returns the
    # first 3 (vote/info/role); decision_mistake sections come later and
    # are truncated. Verify the first three categories in document order
    # plus the cap.
    assert [p.category for p in patterns] == [
        "vote_mistake",
        "info_miss",
        "role_execution",
    ], [p.category for p in patterns]
    # at most 3 returned (cap)
    assert len(patterns) == 3, len(patterns)


def test_extract_llm_mistakes_category_mapping_per_header() -> None:
    cases = {
        "投票错误": "vote_mistake",
        "信息缺失": "info_miss",
        "神职执行": "role_execution",
        "悍跳分析": "decision_mistake",
        "暴露原因": "decision_mistake",
        "角色分工": "decision_mistake",
    }
    for header, expected in cases.items():
        review = f"【{header}】\n- 这是一个足够长的错误描述用于测试\n"
        patterns = ReflectionSynthesizer._extract_llm_mistakes(review, role="seer")
        assert patterns, f"header {header} yielded no patterns"
        assert patterns[0].category == expected, (
            f"header {header} -> {patterns[0].category}, want {expected}"
        )


def test_extract_llm_mistakes_auto_verified_always_false() -> None:
    # CRITICAL safety constraint: LLM mistakes must never bypass the
    # truth-token gate (reflection.py:214).
    patterns = ReflectionSynthesizer._extract_llm_mistakes(_FULL_SIX_SECTIONS, role="seer")
    assert patterns, "expected patterns"
    for p in patterns:
        assert p.auto_verified is False
        assert p.fact_basis == "llm_transferable"
        assert p.corrected_from_llm is False


def test_extract_llm_mistakes_drops_truth_token_bullets() -> None:
    review = """【投票错误】
- 投票时知道实际是狼人还投错
- 没有综合公开票型就匆忙站边
"""
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, role="seer")
    assert len(patterns) == 1, [p.wrong_action for p in patterns]
    assert "实际" not in patterns[0].wrong_action


def test_extract_llm_mistakes_uses_role_default_advice_and_trigger() -> None:
    patterns = ReflectionSynthesizer._extract_llm_mistakes(_FULL_SIX_SECTIONS, role="seer")
    assert patterns
    for p in patterns:
        assert p.better_action == "发言或投票前先核验验人时间线、警徽流和票型承接。", p.better_action
        assert p.trigger  # non-empty


def test_extract_llm_mistakes_empty_input_returns_empty() -> None:
    assert ReflectionSynthesizer._extract_llm_mistakes("", role="seer") == []
    assert ReflectionSynthesizer._extract_llm_mistakes("没有任何 section 的纯文本", role="seer") == []


def test_extract_llm_mistakes_skips_preamble_non_bullet_lines() -> None:
    review = """【投票错误】
本局我做错的地方：
- 第一天投票前没有听完所有发言就站边
"""
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, role="villager")
    assert len(patterns) == 1
    assert "本局我做错" not in patterns[0].wrong_action


def test_extract_llm_mistakes_accepts_bullet_markers() -> None:
    for marker in ("-", "•", "*"):
        review = f"【投票错误】\n{marker} 没有核验票型承接就匆忙投票站边\n"
        patterns = ReflectionSynthesizer._extract_llm_mistakes(review, role="villager")
        assert len(patterns) == 1, f"marker {marker!r} not parsed"


def test_extract_llm_mistakes_short_bullet_dropped() -> None:
    review = "【投票错误】\n- 投错\n- 没有听完所有发言就匆忙站边导致站错\n"
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, role="villager")
    assert len(patterns) == 1
    assert "投错" not in patterns[0].wrong_action


def test_extract_llm_mistakes_scrubs_player_ids() -> None:
    review = "【投票错误】\n- 没有核验p03的警徽流承接就站边p07\n"
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, role="villager")
    assert len(patterns) == 1
    assert "p03" not in patterns[0].wrong_action
    assert "p07" not in patterns[0].wrong_action


# ---------------------------------------------------------------------------
# synthesize merge — LLM mistakes supplement deterministic mistake_patterns
# ---------------------------------------------------------------------------


def test_synthesize_merges_llm_mistakes_after_deterministic() -> None:
    # Deterministic error_analysis produces 1 mistake; the LLM 【投票错误】
    # section adds a second, non-duplicate mistake. Deterministic stays first.
    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="seer",
        faction_won=False,
        error_analysis=["误判某玩家为好人"],
        improvement_suggestions=["投票前先核验警徽流"],
        summary="角色=seer，错误=1项",
    )
    llm_review = (
        "【投票错误】\n"
        "- 没有等预言家报查验就匆忙站边\n"
        "【保留的优点】\n"
        "- 坚持证据优先\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )

    wrong_actions = [p.wrong_action for p in entry.mistake_patterns]
    # deterministic first
    assert "误判某玩家为好人" in wrong_actions[0]
    # LLM bullet supplemented, fact_basis marks provenance
    assert any("报查验" in w for w in wrong_actions)
    llm_added = next(p for p in entry.mistake_patterns if "报查验" in p.wrong_action)
    assert llm_added.fact_basis == "llm_transferable"
    assert llm_added.auto_verified is False


def test_synthesize_llm_mistake_deduped_against_deterministic() -> None:
    # The deterministic wrong_action is near-identical to the LLM bullet;
    # jaccard >= 0.6 must keep the deterministic one and drop the LLM dup.
    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="villager",
        faction_won=False,
        error_analysis=["没有核验票型承接就匆忙投票站边"],
        improvement_suggestions=["先核验票型"],
        summary="角色=villager，错误=1项",
    )
    llm_review = (
        "【投票错误】\n"
        "- 没有核验票型承接就匆忙投票站边\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )

    wrong_actions = [p.wrong_action for p in entry.mistake_patterns]
    # Only one entry — the LLM duplicate must not be appended.
    assert len(wrong_actions) == 1, wrong_actions


def test_synthesize_mistake_total_cap_is_three() -> None:
    # 1 deterministic + plenty of LLM bullets -> total must not exceed 3.
    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="seer",
        faction_won=False,
        error_analysis=["误判某玩家身份"],
        improvement_suggestions=["核验警徽流"],
        summary="角色=seer，错误=1项",
    )
    llm_review = (
        "【投票错误】\n- 没有等预言家报查验就投票\n"
        "【信息缺失】\n- 忽略了警徽流承接\n"
        "【神职执行】\n- 第一晚就浪费了解药\n"
        "【悍跳分析】\n- 被对跳气势唬住\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )

    assert len(entry.mistake_patterns) == 3, [p.wrong_action for p in entry.mistake_patterns]


def test_synthesize_end_to_end_truth_token_bullet_dropped_gate_does_not_reject() -> None:
    # End-to-end: a truth-token bullet in 【投票错误】 is dropped at extraction;
    # a clean bullet is kept; the resulting entry's visible fields carry no
    # truth token, so the quality gate does NOT flag unsafe_truth_claim.
    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="seer",
        faction_won=False,
        successful_strategies=["发言能说明验人心路"],
        improvement_suggestions=["投票前先核验警徽流和票型承接"],
        summary="角色=seer，错误=0项",
    )
    llm_review = (
        "【投票错误】\n"
        "- 投票时知道实际是狼人还投错\n"
        "- 没有综合公开票型就匆忙站边\n"
        "【保留的优点】\n"
        "- 坚持证据优先的站边\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )

    wrong_actions = [p.wrong_action for p in entry.mistake_patterns]
    # truth-token bullet dropped (1b drop still holds as integration check)
    assert not any("实际" in w for w in wrong_actions), wrong_actions
    # clean bullet kept
    assert any("公开票型" in w for w in wrong_actions)

    gated = ReflectionQualityGate().evaluate(entry)
    assert "unsafe_truth_claim" not in gated.quality_flags, gated.quality_flags
    # not hard-rejected on the truth-claim axis
    assert gated.quality_status != ReflectionQualityStatus.REJECTED or "unsafe_truth_claim" not in gated.quality_flags
