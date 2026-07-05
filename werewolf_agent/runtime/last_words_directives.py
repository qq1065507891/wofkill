# -*- coding: utf-8 -*-
"""
构建玩家遗言阶段的策略指令。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.last_words_directives import build_last_words_directive
    >>> build_last_words_directive("seer")
"""

from __future__ import annotations

from typing import Any, Sequence


def build_last_words_directive(player_role: str) -> str:
    """构建通用遗言发言指令。"""
    return (
        f"你已被放逐出局，这是你的遗言。你已确认死亡，身份已公开（{player_role}）。"
        "遗言中你可以：\n"
        "- 预言家：交代你所有的验人结果和警徽流\n"
        "- 猎人：声明你可以开枪（如果被放逐而非毒杀）\n"
        "- 其他好人：表达你对场上局势的最终看法，给存活玩家建议\n"
        "- 狼人：做最后的表演，误导好人\n"
        "遗言必须简短有力。"
    )


def build_hunter_last_words_directive(alive_others: Sequence[str]) -> str:
    """构建猎人被放逐后的遗言提示。"""
    return (
        "你是猎人，被放逐出局。你有权开枪带走一名玩家。\n"
        "遗言中建议：\n"
        "1) 声明猎人身份和开枪意图\n"
        "2) 明确说出你要带走的目标：用'带走{玩家ID}'的格式（如'带走p03'）\n"
        "3) 解释你选择该目标的理由（发言矛盾、站边不明、被查杀等）\n"
        "4) 如果没有明确目标，可以声明'我选择不开枪'\n"
        f"当前存活玩家（不含你）: {list(alive_others)}"
    )


def build_hybrid_last_words_directive() -> str:
    """构建混血儿遗言身份保密提示。"""
    return (
        "【严禁泄漏混血儿身份】你是混血儿，但你的身份不应该在遗言中暴露。\n"
        "遗言中你必须：\n"
        "1) 以普通好人视角发言，绝不提及'主人'、'混血儿'、'阵营选择'等概念\n"
        "2) 不要透露你的主人是谁，也不要暗示你与某位玩家有特殊关系\n"
        "3) 以普通村民身份表达对场上局势的看法和建议\n"
    )


def build_exile_last_words_strategy(
    player_role: str,
    alive_others: Sequence[str],
) -> dict[str, Any]:
    """按身份组合被放逐玩家的遗言策略指令。"""
    strategy_directive: dict[str, Any] = {
        "last_words": build_last_words_directive(player_role),
    }
    if player_role == "hunter":
        strategy_directive["hunter_last_words"] = build_hunter_last_words_directive(
            alive_others
        )
    elif player_role == "hybrid":
        strategy_directive["hybrid_last_words"] = build_hybrid_last_words_directive()
    return strategy_directive
