# -*- coding: utf-8 -*-
"""
构建防御性发言阶段的策略指令和兜底文案。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.defense_speech_directives import build_defense_context_directive
    >>> build_defense_context_directive()
"""

from __future__ import annotations


def build_defense_context_directive() -> str:
    """构建防御性发言的上下文提示。"""
    return (
        "你正处于被质疑/被指控的状态，正在做防御性发言。\n"
        "防御性发言要点：\n"
        "1) 直接回应针对你的具体指控——不能含糊其辞\n"
        "2) 提供你当时发言/投票的合理解释（'我投TA是因为……'）\n"
        "3) 如果指控是误会，提供事实证据（'我可以查我的发言记录'）\n"
        "4) 不要泛泛地喊'我真是好人'——这没有信息量\n"
        "5) 反问指控者的逻辑漏洞（'你为什么认为我有狼面？'）\n"
        "6) 收尾时给出你希望被如何对待的建议（'请听我解释后再投票'）"
    )


def build_empty_defense_speech_fallback(speaker_id: str) -> str:
    """构建空防御性发言的兜底文本。"""
    return (
        f"我是{speaker_id}，我理解大家的质疑。让我解释一下："
        "我当时的判断基于公开信息，可能不全面但绝不是恶意带节奏。"
        "请大家听完我的解释后再做决定。"
    )
