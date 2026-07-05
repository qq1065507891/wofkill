# -*- coding: utf-8 -*-
"""
构建警长竞选发言阶段的策略指令文本。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.sheriff_election_directives import sheriff_uses_seer_protocol
    >>> sheriff_uses_seer_protocol("seer", "")
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from werewolf_agent.core.models import GameState


def sheriff_uses_seer_protocol(player_role: str, wolf_assignment: str) -> bool:
    """判断候选人是否应收到预言家警徽流协议提示。"""
    return player_role == "seer" or wolf_assignment == "fake_seer"


def build_sheriff_badge_flow_instruction(uses_seer_protocol: bool) -> str:
    """构建警徽流私有提示，非预言家协议身份返回空字符串。"""
    if not uses_seer_protocol:
        return ""
    return (
        "2) 你的警徽流：必须留两个晚上的验人对象！"
        "格式如'先验X，后验Y'。如果你验到好人，死后警徽给该好人；"
        "如果验到狼人，则不给警徽（撕徽或给之前验过的好人）。"
        "警徽流是预言家传递信息的核心机制，必须明确留出两夜验人计划。"
    )


def build_sheriff_seer_context(
    public_seer_claimers: set[str],
    *,
    uses_seer_protocol: bool,
) -> str:
    """根据公开预言家声称人数构建竞选局势提示。"""
    if len(public_seer_claimers) >= 2:
        return (
            "场上有多人跳预言家，这是典型的悍跳局面。"
            "真预言家必须坚定立场，用逻辑和验人信息证明自己；"
            "悍跳预言家需要制造合理怀疑，攻击对方的逻辑漏洞。"
        )
    if uses_seer_protocol:
        return (
            "目前警上只有你跳预言家（单边预言家），你的可信度很高。"
            "要充分利用这一点，留下完整的警徽流，让好人信任你。"
        )
    return ""


def collect_previous_sheriff_speeches(
    gs: GameState,
    candidate_id: str,
) -> list[dict[str, str]]:
    """收集当前候选人之前其他候选人的警上发言文本。"""
    speeches: list[dict[str, str]] = []
    for event in gs.events:
        text = event.payload.get("text")
        speaker = event.payload.get("speaker")
        if event.type == "sheriff_speech" and text and speaker != candidate_id:
            speeches.append({
                "speaker": str(speaker),
                "text": str(text),
            })
    return speeches


def build_previous_sheriff_speech_instruction(
    previous_speeches: list[dict[str, str]],
) -> str:
    """构建前人警上发言摘要，提醒候选人避免照搬。"""
    if not previous_speeches:
        return (
            "\n\n你是本轮第一个发言的候选人。"
            "你只能基于目前场上的公开信息发言。"
            "【严禁编造/虚构尚未发言的候选人说过的话】——"
            "你不知道其他候选人会说什么，只能表达自己的立场和分析。"
        )

    prev_summaries = []
    covered_topics = []
    for speech in previous_speeches[-3:]:
        text = speech["text"]
        snippet = text[:150] + ("..." if len(text) > 150 else "")
        prev_summaries.append(f"  [{speech['speaker']}]: {snippet}")
        mentioned = re.findall(r"p\d{2}", text)
        for player_id in mentioned[:3]:
            covered_topics.append(f"{speech['speaker']}已分析过{player_id}")
    prev_texts = "\n".join(prev_summaries)
    covered_note = ""
    if covered_topics:
        covered_note = (
            f"\n【已被覆盖的观点】{', '.join(covered_topics[:6])}。"
            "你必须分析不同的玩家或使用完全不同的推理路径。"
        )
    return (
        f"\n\n【前人发言摘要】在你之前已经有候选人发言了：\n"
        f"{prev_texts}"
        f"{covered_note}\n"
        "你可以反驳，也可以完全忽略前人走自己的分析路线。"
        "【严禁照搬/复述前人发言原文或结构】。"
    )


def build_sheriff_election_speech_directive(
    *,
    style_hint: str,
    task_hint: str,
    badge_flow_instruction: str,
    seer_context: str,
    prev_speech_instruction: str,
    other_candidates: list[str],
) -> dict[str, Any]:
    """聚合警长竞选发言基础指令。"""
    return {
        "sheriff_election_speech": (
            "你正在竞选警长，必须解释上警原因和你的初步判断。"
            "不能只说'我来上警'之类空洞的话，必须有实质内容。"
            "注意：只有预言家（或悍跳预言家）才能留警徽流，其他身份不要提警徽流。"
            "非预言家不要在警上冒充预言家抢警徽。\n"
            f"【你的发言风格】{style_hint}\n"
            f"{task_hint}\n"
            f"{badge_flow_instruction}"
            f"{seer_context}"
            f"{prev_speech_instruction}"
        ),
        "other_candidates": other_candidates,
        "anti_template": (
            "【禁止模板化】你的发言不能机械套用模板。以下句式会让你的发言"
            "被判定为无效：'我这轮先把视角压到XX身上'、'依据是XX最近发言：...'。"
            "你必须有自己独立的角度和分析逻辑。"
        ),
    }


def build_sheriff_role_speech_hint(player_role: str) -> str:
    """按身份构建警上发言重点提示。"""
    role_hints = {
        "idiot": (
            "你是白痴，警上发言重点：观察所有人的发言逻辑，"
            "找出逻辑漏洞或矛盾点，展示你的分析能力。"
        ),
        "hunter": (
            "你是猎人，警上发言重点：关注谁在发言中暴露了信息不对称，"
            "谁的逻辑前后矛盾。你不需要暴露身份。"
        ),
        "witch": (
            "你是女巫，警上发言重点：基于你的夜间信息，"
            "引导讨论方向，但不要暴露你知道的具体信息。"
        ),
        "villager": (
            "你是村民，警上发言重点：用逻辑分析场上信息，"
            "找出预言家真假的判断依据，展示你作为好人的价值。"
        ),
        "hybrid": (
            "你是混血儿，警上发言重点：观察场上局势，"
            "在不确定主人阵营前保持中立分析。"
        ),
    }
    return role_hints.get(player_role, "")


def build_seer_verification_rationale(player_role: str) -> str:
    """预言家专属查验理由要求，其他身份返回空字符串。"""
    if player_role != "seer":
        return ""
    return (
        "【查验理由要求】你每夜的查验目标必须有具体动机。"
        "禁止说'按顺序验'或'随便验的'。正确的说法示例："
        "'N1验p03是因为他发言内容展现了较强的逻辑分析能力，"
        "我需要确认他是好人核心还是狼人伪装'。"
        "查验理由应基于发言内容、投票行为等可观察信息，"
        "不要使用'警上/警下位置'等你在发言时可能记错的信息。"
        "如果没有特殊理由，可以说'首夜随机查验，选择了一个发言量较大的位置'。"
    )


def build_wolf_sheriff_election_directives(
    *,
    wolf_assignment: str,
    wolf_plan: Mapping[str, Any] | None,
    candidate_id: str,
    fake_seer_publicly_claimed: bool,
) -> dict[str, str]:
    """构建狼人警上竞选阶段的私有策略指令。"""
    if wolf_assignment == "fake_seer":
        return {
            "wolf_sheriff_must_claim_seer": (
                "【强制执行】你是团队安排的悍跳预言家！你现在在警上竞选。"
                "你必须在这段发言中跳预言家，报出你的假验人结果和警徽流。"
                "格式参考：'我是预言家，昨晚我验了[玩家]，结果是[好人/查杀]，"
                "我的警徽流是先验[X]后验[Y]。' "
                "不要犹豫、不要含糊——你必须像真预言家一样坚定。"
                "你的假验人结果可以是：金水（假的好人结果）来拉拢人，"
                "或查杀（假的狼人结果）来推好人。选择你认为最优的策略。"
            )
        }
    if not wolf_plan or not wolf_plan.get("fake_seer"):
        return {}

    fake_seer_id = wolf_plan["fake_seer"]
    if fake_seer_id == candidate_id or fake_seer_publicly_claimed:
        return {}
    return {
        "wolf_no_reveal_seer": (
            "【严禁信息穿越】你的队友计划跳预言家但尚未在警上发言。"
            "在你的警上发言中绝不能站边TA或透露TA会跳预言家。"
            "你必须表现得像一个不知道谁是预言家的普通好人。"
            "等TA自己发言后，在后续讨论中你才能像好人一样站边。"
        )
    }
