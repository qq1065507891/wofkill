# -*- coding: utf-8 -*-
"""
构建白天公开发言阶段的通用策略指令和兜底文案。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.day_speech_directives import build_day_speech_base_directive
    >>> build_day_speech_base_directive("")
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_day_speech_base_directive(style_hint: str) -> dict[str, Any]:
    """构建所有白天公开发言共享的基础约束。"""
    return {
        "anti_following_and_peace_night_rule": (
            "不要跟风复述已有指控；如果质疑女巫或预言家，必须给出独立证据并区分事实和推测。"
            "平安夜只代表公开无人死亡，不代表狼人没有刀人。"
            "质疑跳女巫玩家时，应询问是否用药、为什么暂不公开银水、以及发言是否前后矛盾。"
        ),
        "speech_originality": (
            "【发言原创性要求】\n"
            "- 禁止复述其他玩家已经说过的观点——你可以表示同意或反对，但必须补充自己的理由\n"
            "- 禁止使用模板化句式（如'我需要XX正面回应站边、票型'等重复套话）\n"
            "- 你的发言应该展现你独特的思考角度和分析能力\n"
            "- 如果前面已经有人分析了某个玩家，你应换个角度或分析不同的玩家\n"
            "- 【严禁编造任何玩家没有在上方transcript中明确说过的话】"
            "如果某玩家显示'未发表有效言论'或'沉默'，你必须认定该玩家没有做出任何声明、查验或验人报告"
            f"{style_hint}"
        ),
    }


def build_sheriff_speech_directive(
    *,
    is_silenced: bool,
    alive_others: list[str],
) -> dict[str, Any]:
    """构建警长白天发言或沉默时的归票提示。"""
    if is_silenced:
        return {
            "sheriff_silent": (
                "本轮你无法发言，但仍需提交 vote action。"
                "若已提前指定归票目标，在 vote action 的 target_id 字段中给出；"
                "如未指定则由投票开放决定，speech 字段留空。"
            )
        }
    return {
        "sheriff_vote_push": (
            "你是警长，你的发言需要归票：总结本轮讨论的关键信息点，"
            "明确表态你怀疑谁、要推谁，号召大家集中投票。"
            "警长归票是核心职责，不能含糊其辞。"
        ),
        "sheriff_alive_others": alive_others,
    }


def build_torn_badge_speech_state() -> str:
    """构建撕徽后无警长状态提示。"""
    return "本局无警长；本轮发言顺序随机；无归票人。"


def collect_sheriff_election_speeches(gs: GameState) -> list[dict[str, str]]:
    """收集警上竞选阶段的有效发言。"""
    speeches: list[dict[str, str]] = []
    for event in gs.events:
        text = event.payload.get("text")
        if event.type == "sheriff_speech" and text:
            speeches.append({
                "speaker": str(event.payload.get("speaker", "")),
                "text": str(text),
            })
    return speeches


def build_sheriff_election_record(
    sheriff_speeches: list[dict[str, str]],
) -> str:
    """把警上竞选发言压缩为白天讨论可用摘要。"""
    if not sheriff_speeches:
        return ""
    speech_summaries = []
    for speech in sheriff_speeches:
        text = speech["text"]
        snippet = text[:120] + ("..." if len(text) > 120 else "")
        speech_summaries.append(f"  [{speech['speaker']}]: {snippet}")
    return (
        "以下是警上竞选环节各候选人发言的摘要：\n"
        + "\n".join(speech_summaries)
    )


def build_empty_day_speech_fallback(speaker_id: str, target_hint: str) -> str:
    """构建普通空白天发言的兜底文本。"""
    return (
        f"我是{speaker_id}，我认为目前场上信息不够明确。"
        f"我关注{target_hint}的发言，需要更多信息来判断。"
    )


def build_sanitized_seer_claim_fallback(
    speaker_id: str,
    target_hint: str,
) -> str:
    """构建预言家宣称违规后的安全兜底文本。"""
    return (
        f"我是{speaker_id}，目前信息不足，我需要先观察其他玩家的发言再做判断。"
        f"我会重点关注{target_hint}的站边和投票倾向。"
    )
