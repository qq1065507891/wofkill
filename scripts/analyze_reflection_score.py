# -*- coding: utf-8 -*-
"""
分析 ReflectionQualityGate 对模拟复盘内容的评分结果。

作者: Project contributors
修改日期: 2026-07-07

使用示例:
    >>> import scripts.analyze_reflection_score
    >>> scripts.analyze_reflection_score.PLUS_ITEMS
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from werewolf_agent.memory.reflection import (
    ReflectionQualityGate,
    ReflectionSynthesizer,
)
from werewolf_agent.memory.schemas import ReviewReport

# +项权重(来自 evaluate)与触发条件说明,供分解表引用。
PLUS_ITEMS = [
    ("mistake_patterns(+0.25)", "任意 pattern trigger&better_action 非空"),
    ("preserved_strengths(+0.15)", "任意 strength reuse_condition 非空"),
    ("complete_prompt_card(+0.25)", "theme+lesson+trigger_signals+action+misuse 齐全"),
    ("situation_signature(+0.10)", "role 非空 且 (phase_focus 或 game_patterns)"),
    ("auto_verified/llm_transferable(+0.10)", "prompt_card.auto_verified 或 fact_basis==llm_transferable"),
    ("actionable_advice(+0.10)", "任一 advice 含 先/不要/避免/必须/优先/核验/比较/列"),
    ("role_specific(+0.05)", "visible_blob 含 角色关键词"),
]
MINUS_ITEMS = [
    ("generic_text(-0.25)", "含泛化套话短语"),
    ("short_prompt_card(-0.15)", "prompt_card 内容 < 80 字"),
    ("missing_trigger(-0.15)", "phase_focus/game_patterns/trigger_signals 全空"),
    ("duplicate(-0.20)", "与已 approved 条目 jaccard>=0.70"),
    ("source_truncated(-0.05)", "llm_self_review+auto_review_summary > 1600 字"),
]


# ---- 三种代表性 LLM self-review(模拟真实输出) ---------------------------

# 乐观:每条 bullet 30-50 字,含多个 section。
OPTIMISTIC_REVIEW = """【投票错误】
- 第二天白天没有等待预言家报验人结果就站边8号，错过对跳信息导致好人分票。
- 第三天听了8号跳预言家却信了9号的煽动发言，投错了关键一票把真预言家放逐。

【信息缺失】
- 忽略了警徽流这一关键信号，没把两轮验人时间线串起来交叉核验。
- 没有比较两派发言的逻辑承接，只凭情感倾向就做了判断。

【保留的优点】
- 第一天发言时先列出已知公开事实再下结论，逻辑链条清晰可复盘。
- 投票前核验了票型承接关系，避免了被深水狼带节奏冲票。"""

# 典型:每条 bullet 15-25 字。
TYPICAL_REVIEW = """【投票错误】
- 第二天没等验人结果就站边，好人分票。
- 第三天被9号煽动发言误导，投错关键票。

【信息缺失】
- 忽略警徽流信号，验人时间线没串起来。
- 没比较两派发言逻辑承接。

【保留的优点】
- 发言前列公开事实再下结论。
- 投票前核验票型承接。"""

# 悲观:每条 bullet 6-10 字,接近 6 字下限。
PESSIMISTIC_REVIEW = """【投票错误】
- 没等验人就站边。
- 被煽动发言误导。

【信息缺失】
- 忽略警徽流。
- 没比较发言逻辑。

【保留的优点】
- 发言前列事实。
- 投票前核验票型。"""

# 无结构:LLM 只输出一句话,无 【...】 section,parser 抽不出 mistake。
# 模拟数据库里 51 字那种极短输出 —— plan Risks 担忧的真正卡点。
NO_SECTION_REVIEW = "这局没什么问题，下局继续努力，总结经验。"

# 无结构且非套话:中等长度但无 section header。
NO_SECTION_V2_REVIEW = (
    "这局站边失误，听了煽动发言投错票，下局要先核验证据再判断。"
)


def make_report(
    role: str = "seer",
    won: bool = False,
    deterministic_mistakes: bool = False,
) -> ReviewReport:
    """ReviewReport 在 1b 路径下由 deterministic review 产出。

    deterministic_mistakes=False 是 plan Risks 担忧的场景:确定性 review
    没抓到 mistake,反思质量完全依赖 LLM self-review 的 parser 抽取。
    =True 是 defect B 对照:确定性 review 注入 mistake(如
    "误判X为Y，最佳角色概率0.80" 这类较长的确定性结论)。
    """
    error_analysis = (
        ["误判9号为预言家，实际为狼人，最佳角色概率0.80"]
        if deterministic_mistakes
        else []
    )
    return ReviewReport(
        game_id="g_analysis_001",
        player_id="p1",
        role=role,
        faction_won=won,
        error_analysis=error_analysis,
        successful_strategies=["发言前列出公开事实再下结论，票型承接关系核验到位"],
        improvement_suggestions=[
            "投票或站边前先核验验人时间线、警徽流和票型承接。",
            "比较对跳双方发言的逻辑链条，避免凭情感判断。",
        ],
        deceived_by=[],
        summary="复盘 D2 站边失误。",
    )


def run_scenario(
    name: str,
    review: str,
    role: str,
    won: bool,
    deterministic_mistakes: bool = False,
) -> dict:
    report = make_report(
        role=role, won=won, deterministic_mistakes=deterministic_mistakes
    )
    synthesizer = ReflectionSynthesizer()
    entry = synthesizer.synthesize(
        llm_self_review=review,
        review_report=report,
        faction="good",
    )
    # gate 无历史 -> 不触发 duplicate。
    scored = ReflectionQualityGate().evaluate(entry)
    card_len = len(
        scored.prompt_card.theme
        + scored.prompt_card.lesson
        + "".join(scored.prompt_card.trigger_signals)
        + scored.prompt_card.recommended_action
        + scored.prompt_card.misuse_risk
    )
    return {
        "name": name,
        "score": scored.quality_score,
        "status": scored.quality_status.value,
        "flags": scored.quality_flags,
        "n_mistakes": len(scored.mistake_patterns),
        "n_strengths": len(scored.preserved_strengths),
        "card_len": card_len,
        "fact_basis": scored.prompt_card.fact_basis,
        "auto_verified": scored.prompt_card.auto_verified,
        "trigger_signals": scored.prompt_card.trigger_signals,
        "phase_focus": scored.situation_signature.phase_focus,
        "game_patterns": scored.situation_signature.game_patterns,
        "mistake_sources": [
            f"{m.fact_basis}/{'auto' if m.auto_verified else 'llm'}"
            for m in scored.mistake_patterns
        ],
        "review_chars": len(review),
    }


def print_plus_minus_legend() -> None:
    print("=== 评分项(+/-)权重速查 ===")
    for label, cond in PLUS_ITEMS:
        print(f"  {label:42s}  {cond}")
    for label, cond in MINUS_ITEMS:
        print(f"  {label:42s}  {cond}")
    print()


def print_table(rows: list[dict]) -> None:
    print("=== Score 分解表(1b 路径:deterministic error_analysis 为空) ===")
    header = (
        f"{'场景':<10}{'字数':>5}{'score':>7}{'status':<13}"
        f"{'mistakes':>9}{'strengths':>10}{'card_len':>9}  flags"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        flags_str = ",".join(r["flags"]) or "-"
        print(
            f"{r['name']:<10}{r['review_chars']:>5}{r['score']:>7.2f}"
            f"{r['status']:<13}{r['n_mistakes']:>9}{r['n_strengths']:>10}"
            f"{r['card_len']:>9}  {flags_str}"
        )
    print()
    for r in rows:
        print(
            f"  [{r['name']}] mistake_sources={r['mistake_sources']} "
            f"fact_basis={r['fact_basis']} auto_verified={r['auto_verified']} "
            f"trigger_signals={r['trigger_signals']} "
            f"phase_focus={r['phase_focus']} game_patterns={r['game_patterns']}"
        )
    print()


def main() -> None:
    print_plus_minus_legend()
    rows = [
        run_scenario("乐观", OPTIMISTIC_REVIEW, role="seer", won=False),
        run_scenario("典型", TYPICAL_REVIEW, role="seer", won=False),
        run_scenario("悲观", PESSIMISTIC_REVIEW, role="seer", won=False),
        run_scenario("无section", NO_SECTION_REVIEW, role="seer", won=False),
        run_scenario(
            "无section+defB",
            NO_SECTION_V2_REVIEW,
            role="seer",
            won=False,
            deterministic_mistakes=True,
        ),
    ]
    print_table(rows)

    approved = [r for r in rows if r["status"] == "approved"]
    print("=== 结论 ===")
    print(f"approved(>=0.70): {[r['name'] for r in approved]}")
    print(f"review_only(0.40-0.69): {[r['name'] for r in rows if r['status']=='review_only']}")
    print(f"rejected(<0.40): {[r['name'] for r in rows if r['status']=='rejected']}")
    for r in rows:
        # 手算 +项总和,验证 score 来源。
        plus = 0.0
        plus += 0.25 if r["n_mistakes"] else 0.0
        plus += 0.15 if r["n_strengths"] else 0.0
        plus += 0.25  # complete_prompt_card:synthesizer 永远填齐
        plus += 0.10  # situation_signature:phase_focus 非空
        plus += 0.10  # fact_basis==llm_transferable 或 auto_verified
        plus += 0.10  # actionable_advice:default advice 含 核验/先/比较
        plus += 0.05  # role_specific:seer 关键词(预言家/警徽流/对跳)
        minus = 0.0
        for f in r["flags"]:
            if f == "generic_text":
                minus += 0.25
            elif f == "short_prompt_card":
                minus += 0.15
            elif f == "missing_trigger":
                minus += 0.15
            elif f == "source_truncated":
                minus += 0.05
        calc = round(max(0.0, min(1.0, plus + minus)), 2)
        print(
            f"  [{r['name']}] +项合计={plus:.2f} -项合计={minus:.2f} "
            f"-> 计算score={calc:.2f} 实际score={r['score']:.2f} "
            f"{'OK' if calc == r['score'] else 'MISMATCH'}"
        )


if __name__ == "__main__":
    main()
