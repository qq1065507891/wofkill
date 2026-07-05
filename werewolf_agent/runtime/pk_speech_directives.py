# -*- coding: utf-8 -*-
"""
构建 PK 发言阶段的身份策略指令。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.pk_speech_directives import build_pk_speech_strategy
    >>> build_pk_speech_strategy(...)
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.core.models import GameState


logger = logging.getLogger(__name__)


def build_pk_speech_strategy(gs: GameState, speaker_id: str) -> dict[str, Any]:
    """根据发言者身份构建 PK 发言阶段的策略指令。"""
    pk_strategy: dict[str, Any] = {
        "pk_urgent": (
            "你正处于PK发言阶段——平票候选人只有一次发言机会，必须在这一轮内说服足够多的人改投你。"
            "不要再'等下一轮'，不要再'观察'，直接亮出你最强的证据或分析。"
        ),
    }
    player_role = gs.players[speaker_id].role if speaker_id in gs.players else ""
    if player_role == "werewolf":
        pk_strategy["wolf_pk_push"] = (
            "你是狼人，PK发言策略：\n"
            "1) 攻击对手的发言漏洞——TA的逻辑不完整、TA的站边前后矛盾\n"
            "2) 表现得像一个有分析能力的好人，不要替狼队说话\n"
            "3) 如果场上有队友的推人目标，借机把票型引导到目标玩家"
        )
    elif player_role == "seer":
        _add_seer_pk_check_evidence(pk_strategy, gs)
    elif player_role == "witch":
        pk_strategy["witch_pk_evidence"] = (
            "你是女巫，PK发言策略：\n"
            "1) 不要轻易透露药水状态，但你需要给出可信的分析来赢得PK\n"
            "2) 引用场上具体的发言矛盾、票型异常来支撑你的判断\n"
            "3) 如果你救了某人（银水），可以暗示'我手里有信息'但不要明说"
        )
    elif player_role == "hunter":
        pk_strategy["hunter_pk_pressure"] = (
            "你是猎人，PK发言策略：\n"
            "1) 利用'我有枪'的威慑——明确说'我被放逐会开枪带走最可疑的人'\n"
            "2) 这会给狼队压力，让他们考虑放逐你的风险\n"
            "3) 但不要虚张声势说已经决定带谁"
        )
    elif player_role == "villager":
        pk_strategy["villager_pk_logic"] = (
            "你是普通村民，PK发言必须基于公开信息逻辑分析：\n"
            "1) 引用场上具体的发言矛盾、票型异常、警徽流\n"
            "2) 不要喊'我是好人'——这没有信息量\n"
            "3) 直接分析对手为什么更像狼，列出2-3个具体证据"
        )
    elif player_role == "idiot":
        pk_strategy["idiot_pk_caution"] = (
            "你是白痴，PK发言：\n"
            "1) 翻牌前的白痴不要暴露身份，专注逻辑分析\n"
            "2) 如果被投出后翻牌，只能发表遗言，随后出局，不再参与后续PK"
        )
    elif player_role == "hybrid":
        master_id = gs.hybrid_master_id
        if master_id:
            pk_strategy["hybrid_pk_master_align"] = (
                f"你是混血儿，主人是{master_id}。"
                "PK发言要表现得像主人的判断方向——"
                "如果主人在场，分析与主人站边一致；"
                "但不要每轮都跟主人保持完全一致，那会暴露关系。"
            )
    return pk_strategy


def _add_seer_pk_check_evidence(pk_strategy: dict[str, Any], gs: GameState) -> None:
    """把预言家已有查验结果计入 PK 发言策略。"""
    try:
        check_results = []
        for event in gs.events:
            if event.type == "seer_check":
                check_results.append({
                    "target": event.payload["target_id"],
                    "alignment": event.payload["alignment"],
                    "night": event.payload["night_number"],
                })
        if check_results:
            pk_strategy["seer_pk_check_evidence"] = (
                "你是预言家，PK发言必须以你的查验结果为核心："
                f"你已获得 {len(check_results)} 个查验结果。"
                "在PK中直接报出最关键的一个查杀或金水，"
                "告诉所有人'信我，我查了[玩家]是[好人/狼人]'，"
                "让对跳预言家或你的对手无法在60秒内反驳。"
            )
    except Exception:
        logger.debug("Failed to build seer PK check evidence", exc_info=True)
