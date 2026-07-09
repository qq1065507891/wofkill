# -*- coding: utf-8 -*-
"""
构建白天投票阶段的策略指令文本。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.runtime.day_vote_directives import build_day_vote_base_directive
    >>> build_day_vote_base_directive("villager", allow_abstain=False, consecutive_no_exile=0)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState

_GOOD_VOTE_ROLES = frozenset({"villager", "seer", "witch", "hunter", "idiot"})


def build_day_vote_base_directive(
    voter_role: str,
    *,
    allow_abstain: bool,
    consecutive_no_exile: int,
) -> dict[str, Any]:
    """构建所有投票角色共享的结构化投票契约和基础压力。"""
    directive: dict[str, Any] = {
        "require_vote_quality": True,
        "vote_structured_contract": {
            "seer_stance": ["trust", "distrust", "undecided", "no_claim"],
            "vote_basis": [
                "seer_check",
                "seer_siding",
                "speech_logic",
                "vote_pattern",
                "pressure_test",
                "anti_herd",
                "fallback",
            ],
        },
        "vote_silent": (
            "投票阶段不允许公开发言。speech字段必须留空。"
            "你只能内心选择要投谁，不能在投票时发表任何公开言论。"
            "请在reason字段中写下简短公开理由；同时在JSON中额外写"
            "seer_stance、vote_basis、standing_with_seer、suspect_reason、"
            "not_voting_reason、candidate_comparison、private_reason。"
            "这些字段是你的投票心理活动，只给主持人审计，不会公开给其他玩家。"
        ),
        "vote_strategy": (
            "投票原则：\n"
            "1) 有查杀走查杀：如果被信任的预言家查杀了某人，优先投查杀对象。"
            "2) 跟预言家走：听完发言后，根据你信任的预言家的归票方向投票。"
            "3) 警上单边预言家可信度高，警下跳预言家的可信度很低。"
            "4) 如果没有明确查杀，投发言最可疑、逻辑最不通的人。"
        ),
    }
    if voter_role in _GOOD_VOTE_ROLES:
        directive["good_vote_decision_guard"] = (
            "【好人投票决策纪律】先按硬信息优先级排序，再做选择："
            "1) 预言家查验/可信查杀；2) 已知死亡身份和死亡原因；"
            "3) 投票票型；4) 警徽流/警长归票；5) 发言怀疑。"
            "【出错成本】投票前必须比较：如果目标其实是预言家、女巫、猎人、白痴或关键金水，"
            "今天出错会不会直接导致屠神/屠民；若只有发言风格可疑而没有查验或强票型，"
            "不要轻易推出疑似神职或高价值好人。"
        )
    if not allow_abstain:
        pressure_parts = ["必须投票选出一名玩家放逐，不能弃票。"]
        if consecutive_no_exile > 0:
            pressure_parts.append(f"已经连续{consecutive_no_exile}天无人出局，必须做出决定。")
        directive["vote_pressure"] = " ".join(pressure_parts)
    return directive


def build_vote_anti_herd_directive() -> str:
    """构建通用防机械跟票提示。"""
    return (
        "Do not mechanically follow a near-unanimous push unless it has "
        "concrete evidence such as checks, counterclaims, vote records, "
        "contradictions, or quoted speeches."
    )


def build_fallback_seer_vote_strategy() -> str:
    """构建预言家投票策略生成失败时的兜底提示。"""
    return "你是预言家，投票时以你的查验结果为核心依据。"


def build_hybrid_vote_strategy(master_id: str) -> str:
    """构建混血儿投票策略。"""
    return (
        f"你是混血儿，主人是 {master_id}。投票策略：\n"
        "1) 观察主人投谁——如果主人投了某人，考虑也投该方向（对主人阵营有利）\n"
        "2) 不要每轮都跟主人投一样的人，那会暴露你们的关系\n"
        "3) 如果主人投的人你确实认为可疑，正常投票即可\n"
        "4) 如果主人被投了——你需要判断主人被放逐是否对主人阵营不利，"
        "考虑是否投别人来稀释票数\n"
        "5) 绝对不要在投票理由中暴露你的混血儿身份"
    )


def build_seer_vote_strategy(gs: GameState) -> str:
    """根据预言家查验事件构建预言家投票策略。"""
    check_results = []
    for event in gs.events:
        if event.type == "seer_check":
            check_results.append({
                "target": event.payload["target_id"],
                "alignment": event.payload["alignment"],
                "night": event.payload["night_number"],
            })
    wolf_checks = [item for item in check_results if item["alignment"] == "wolf"]
    good_checks = [item for item in check_results if item["alignment"] == "good"]
    wolf_list = "、".join(item["target"] for item in wolf_checks) or "（无）"
    good_list = "、".join(item["target"] for item in good_checks) or "（无）"
    return (
        "你是预言家，投票策略：\n"
        f"1) 你已查验出狼人: {wolf_list}——必须把票投给这些查杀对象中的某一个\n"
        f"2) 你已查验出好人: {good_list}——不要投这些人\n"
        "3) 如果场上多个人被查杀，优先投你最近查杀、警徽流计划中的下一个\n"
        "4) 如果场上没有查杀对象，引用票型/警徽流/发言矛盾\n"
        "5) 投票时公开重申你的查杀——这是预言家的核心职责"
    )


def build_witch_vote_strategy() -> str:
    """构建女巫投票策略。"""
    return (
        "你是女巫，投票策略：\n"
        "1) 你的解药目标（银水）是好人的强信号——给银水站台、帮其站边\n"
        "2) 你的毒药目标如果是狼人，那一票已经定局；如果是好人，提醒自己不要把票投给无辜者\n"
        "3) 不要在公开投票理由中提及药水使用细节（'我救了TA'/'我毒了TA'）\n"
        "4) 但你可以引用场上其他公开信息（发言矛盾、票型）来支撑你的投票"
    )


def build_hunter_vote_strategy() -> str:
    """构建猎人投票策略。"""
    return (
        "你是猎人，投票策略：\n"
        "1) 投完票后可能被放逐——一旦你被放逐，你会开枪\n"
        "2) 投票时考虑：如果我被放逐，我最想带谁？把票投给最像狼的人\n"
        "3) 不要投给明显是好人的人——浪费你的枪\n"
        "4) 如果你不想暴露自己，仍应基于公开证据独立选择最可疑目标，不机械跟随多数"
    )


def build_villager_vote_strategy(seer_claimants: set[str]) -> str:
    """根据预言家公开声称者数量构建村民和白痴投票策略。"""
    if len(seer_claimants) == 1:
        claimant = sorted(seer_claimants)[0]
        return (
            "你是普通好人（无私有信息），投票策略必须基于公开信息独立判断：\n"
            f"1) 场上只有{claimant}单边跳预言家（无对跳预言家）——"
            "单边预言家的可信度较高，可以优先跟其查杀走\n"
            "2) 但即使是单边预言家，也要看TA的发言是否有验人逻辑链、是否遵守警徽流\n"
            "3) 不要无条件跟任何人的票——先用你自己的分析判断谁更像狼\n"
            "4) 关注票型数据：谁在保谁、谁在投谁——狼人倾向于抱团投票\n"
            "5) 如果场上没有查杀，投发言逻辑最混乱、站边最模糊的人\n"
            "6) 不要投自己——这没有任何价值"
        )
    return (
        "你是普通好人（无私有信息），投票策略必须基于公开信息独立判断：\n"
        "1) 先判断预言家真假：\n"
        "   - 单边预言家（无对跳）：可信度高，可以跟其查杀走\n"
        f"   - 对跳预言家 {sorted(seer_claimants) if seer_claimants else '（暂无）'}："
        "谁的验人逻辑链更完整？谁的发言有实质信息？谁在遵守警徽流？\n"
        "2) 不要无条件跟任何人的票——先用你自己的分析判断谁更像狼\n"
        "3) 关注票型数据：谁在保谁、谁在投谁——狼人倾向于抱团投票\n"
        "4) 如果场上没有查杀，投发言逻辑最混乱、站边最模糊的人\n"
        "5) 不要投自己——这没有任何价值"
    )
