# -*- coding: utf-8 -*-
"""
按角色阵营生成赛后复盘提示词。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.runtime.reflection_prompt import build_reflection_prompt
    >>> build_reflection_prompt(player, winner="good", hybrid_master_faction=None)
"""

from __future__ import annotations

from typing import Any


GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}


def build_reflection_prompt(
    player: Any,
    winner: str,
    hybrid_master_faction: str | None,
) -> str:
    """Build the role-family-specific post-game reflection prompt."""
    role = (player.role if player else "") or ""
    faction_result = "胜" if (
        (role in GOOD_ROLES and winner == "good")
        or (role == "werewolf" and winner == "werewolf")
        or (role == "hybrid" and (
            (hybrid_master_faction == "good" and winner == "good")
            or (hybrid_master_faction == "werewolf" and winner == "werewolf")
        ))
    ) else "负"

    if role in GOOD_ROLES:
        return GOOD_REFLECTION_TEMPLATE.format(
            faction_result=faction_result,
            role=role,
        )
    if role == "werewolf":
        return WOLF_REFLECTION_TEMPLATE.format(faction_result=faction_result)
    if role == "hybrid":
        if hybrid_master_faction == "good":
            return GOOD_REFLECTION_TEMPLATE.format(
                faction_result=faction_result,
                role="hybrid(跟好人)",
            )
        if hybrid_master_faction == "werewolf":
            return WOLF_REFLECTION_TEMPLATE.format(faction_result=faction_result)
    return GENERIC_REFLECTION_TEMPLATE.format(
        faction_result=faction_result,
        role=role,
    )


GOOD_REFLECTION_TEMPLATE = """你是{role},本局好人阵营{faction_result}。请按以下结构复盘:

【投票错误】本局你投过谁?有没有推错人?为什么站错边?
- 具体指出哪一天的投票决策有误,错投了谁,该投谁
- 分析站错边的根因(信息不足/被悍跳狼误导/被情绪带动)

【信息缺失】哪些关键信号被你忽略了?
- 预言家的查杀声明 / 悍跳狼的逻辑漏洞 / 票型异常(分票/跟票)
- 女巫的解药用错 / 毒药空过 / 白痴翻牌时机

【神职执行】(仅神职需要回答)
- 预言家:警徽流是否清晰?是否被首推?
- 女巫:解药救了谁?是否值得?毒药目标对了吗?
- 猎人:被放逐/夜杀时是否开枪?目标对了吗?
- 白痴:翻牌时机是否合适?

【保留的优点】本局你做对了什么?必须列出 1-2 个具体策略,下局复用:
- 例如:"N2 我用解药救了警长,后续警长归票带我们翻盘"
- 例如:"我在 D3 提前质疑悍跳狼的警徽流时间线,被采信了"

【PII】不要写具体玩家 ID,用"某玩家"代替(后处理会进一步脱敏)。

"""


WOLF_REFLECTION_TEMPLATE = """你是狼人,本局狼队{faction_result}。请按以下结构复盘:

【悍跳分析】(如果有狼跳预言家)
- 悍跳发言为什么没人信?逻辑漏洞在哪?
- 验人口径是否前后矛盾?警徽流是否清晰?
- 真预言家对跳后,悍跳狼是否被迅速识别?

【暴露原因】狼队为什么被识破?
- 哪些发言/票型留下了痕迹(白天跟票太齐/发言风格雷同/悍跳失误)?
- 哪一局开始局势不可逆?转折点是什么?

【角色分工】深水/冲锋/倒钩的执行:
- 深水狼:是否成功藏到最后?有没有过早暴露?
- 冲锋狼:为悍跳狼站台是否有效?是否用力过猛?
- 倒钩狼:踩队友获取信任是否成功?

【保留的优点】本局你做对了什么?必须列出 1-2 个具体策略,下局复用:
- 例如:"我们 N1 空刀让好人视野混乱,第二天悍跳狼拿到警徽"
- 例如:"倒钩狼 D3 故意踩悍跳队友,后期反水一击致命"

【PII】不要写本局具体玩家 ID(好人的真实身份),用"预言家"/"女巫"/"被查杀的神职"代替(后处理会进一步脱敏)。

"""


GENERIC_REFLECTION_TEMPLATE = """你是{role},本局{faction_result}。请复盘:
- 本局你做了哪些关键判断?哪些对?哪些错?
- 有没有被谁欺骗或误导?下局如何改进?
- 【保留的优点】本局你做对了什么?必须列出 1-2 个具体策略,下局复用。
"""
