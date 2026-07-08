# -*- coding: utf-8 -*-
"""
玩家 fallback 发言构造 helper。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.player_fallback_speech import build_fallback_speech
    >>> build_fallback_speech(context)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from werewolf_agent.agents.schemas import AgentContext, TaskType

logger = logging.getLogger(__name__)

_FALLBACK_PREFIX = "[FALLBACK]"

_WOLF_DISCUSSION_TARGET_TEMPLATES = (
    f"{_FALLBACK_PREFIX}狼队夜聊我建议优先刀{{target}}，这个位置如果是神职能压缩好人信息。",
    f"{_FALLBACK_PREFIX}狼队视角我倾向先处理{{target}}，今晚统一刀口，明天白天把压力转出去。",
    f"{_FALLBACK_PREFIX}我建议本轮刀{{target}}，后续容易被好人认证，统一行动再分配冲锋倒钩。",
    f"{_FALLBACK_PREFIX}从狼队视角看{{target}}威胁较大，建议集中票型统一处理，避免分刀。",
    f"{_FALLBACK_PREFIX}今晚目标{{target}}，理由是这个位置存活越久越难处理，尽早解决。",
    f"{_FALLBACK_PREFIX}狼队今晚刀{{target}}，明天我们安排一人引领讨论方向，一人补位配合。",
    f"{_FALLBACK_PREFIX}我倾向于刀{{target}}，它在神职概率较高的位置，赌中收益很大。",
)
_WOLF_DISCUSSION_TEMPLATES = (
    f"{_FALLBACK_PREFIX}狼队夜聊先统一刀口，再分配明天的冲锋位和倒钩位，避免发言互相打架。",
    f"{_FALLBACK_PREFIX}狼队视角今晚先别分散意见，优先找神职或强势带队位，明天顺着信息推人。",
    f"{_FALLBACK_PREFIX}建议先整理每人明天的站位：一人带节奏，一人补逻辑，一人适度倒钩保护团队。",
    f"{_FALLBACK_PREFIX}今晚不宜空刀——连续空刀会暴露战术意图，至少制造一个刀口给女巫压力。",
    f"{_FALLBACK_PREFIX}狼队需要确定今晚行动，刀口一致才能最大化信息不对称优势。",
    f"{_FALLBACK_PREFIX}提醒队友注意发言一致性，不同人的站边不要互相矛盾以免被好人抓住破绽。",
    f"{_FALLBACK_PREFIX}狼队视角先定今晚目标，再决定明天谁冲锋谁潜伏，分工明确胜率更高。",
)
_SHERIFF_OR_PK_TEMPLATES = (
    f"{_FALLBACK_PREFIX}我上警是想给出自己的独立判断视角，重点关注前几位发言的逻辑一致性。",
    f"{_FALLBACK_PREFIX}我参加警长竞选，希望通过观察和提问帮好人理清局势。",
    f"{_FALLBACK_PREFIX}上警是为了确保好人阵营有人能带节奏，我会根据后续发言调整站边。",
    f"{_FALLBACK_PREFIX}我是好人视角上警，主要是防止狼人控场，请大家根据发言质量判断。",
    f"{_FALLBACK_PREFIX}上警竞选，我有信心带队——我会认真分析每个人的发言和投票逻辑。",
    f"{_FALLBACK_PREFIX}参选警长不是为了秀存在感，而是要让好人阵营有一个清晰的发言方向。",
    f"{_FALLBACK_PREFIX}我上警是对局势负责，不想看到警徽落入可疑玩家手中。",
)
_DEFENSE_TEMPLATES = (
    f"{_FALLBACK_PREFIX}我确实不是狼人，请大家仔细分析我的发言和投票逻辑。",
    f"{_FALLBACK_PREFIX}我没有理由被推，关注我的人应该先看看自己的视角是否正确。",
    f"{_FALLBACK_PREFIX}我是好人，我的选择都是基于公开信息，没有任何隐藏动机。",
    f"{_FALLBACK_PREFIX}回顾我的发言和投票，没有任何矛盾之处，被推可能是狼人在带节奏。",
    f"{_FALLBACK_PREFIX}如果你们仔细看我的逻辑链，会发现我的站边和推理都是连贯且合理的。",
    f"{_FALLBACK_PREFIX}被质疑很正常，但我希望大家关注推我的人背后的动机——可能是狼人抗推。",
    f"{_FALLBACK_PREFIX}请好人看清局势，我不是狼，真正的问题可能在那些急于归票的人身上。",
)
_LAST_WORDS_TARGET_TEMPLATES = (
    f"{_FALLBACK_PREFIX}遗言重点关注{{target}}，发言逻辑存在明显矛盾，请大家后续留意。",
    f"{_FALLBACK_PREFIX}走了，提醒大家注意{{target}}的立场和行为不一致，我对此有较大疑虑。",
    f"{_FALLBACK_PREFIX}最后说一句，{{target}}的发言中有些关键点没有解释清楚，值得深挖。",
    f"{_FALLBACK_PREFIX}遗言不多说，但{{target}}的投票路线和发言立场严重不符，建议重点观察。",
    f"{_FALLBACK_PREFIX}我注意到{{target}}在关键轮次的站边突变，这不正常——好人阵营请留意。",
    f"{_FALLBACK_PREFIX}临别前提醒一句：{{target}}可能是突破口，其逻辑链有明显断裂。",
    f"{_FALLBACK_PREFIX}遗言：关注{{target}}，其行为模式与好人视角不符，建议后续深入盘查。",
)
_LAST_WORDS_TEMPLATES = (
    f"{_FALLBACK_PREFIX}遗言不多说了，请大家仔细分析每个人的站边逻辑和投票记录。",
    f"{_FALLBACK_PREFIX}我相信好人阵营能通过票型和发言找出狼人，加油。",
    f"{_FALLBACK_PREFIX}最后提醒一下，注意观察谁在关键投票中立场摇摆。",
    f"{_FALLBACK_PREFIX}遗言：希望大家冷静分析，不要被情绪化发言带偏，聚焦票型和逻辑链。",
    f"{_FALLBACK_PREFIX}走了。好人阵营请重点复盘关键轮的投票分布，那里有答案。",
    f"{_FALLBACK_PREFIX}我的身份是好人，希望我的出局能让你们更清晰地看清局势。",
    f"{_FALLBACK_PREFIX}遗言简短：信任逻辑不信任直觉，仔细对比每个人的发言与投票是否一致。",
)
_DAY_TARGET_TEMPLATES = (
    f"{_FALLBACK_PREFIX}我目前对{{target}}有较大疑虑，其发言逻辑不够连贯，需要进一步观察。",
    f"{_FALLBACK_PREFIX}从现有信息来看，{{target}}的立场和行为有矛盾，我倾向于关注这个方向。",
    f"{_FALLBACK_PREFIX}我分析了一下，{{target}}的发言中有些观点缺乏依据，我对此保持警惕。",
    f"{_FALLBACK_PREFIX}综合来看{{target}}在关键节点的表现比较可疑，值得进一步深挖其动机。",
    f"{_FALLBACK_PREFIX}{{target}}的投票行为和发言内容之间存在落差，这一点不太对劲。",
    f"{_FALLBACK_PREFIX}目前关注{{target}}——其逻辑推理链中有几处跳跃，不像自然的好人思维。",
    f"{_FALLBACK_PREFIX}{{target}}的站边轨迹值得关注，在关键轮次的变化缺乏充分解释。",
)
_DAY_TEMPLATES = (
    f"{_FALLBACK_PREFIX}我目前还在整理信息，请大家注意分析发言中的逻辑矛盾和票型走向。",
    f"{_FALLBACK_PREFIX}暂时没有确定的目标，但我会重点关注后续发言中立场摇摆的人。",
    f"{_FALLBACK_PREFIX}根据现有公开信息，我建议大家都仔细梳理一下各人的站边逻辑。",
    f"{_FALLBACK_PREFIX}这一轮信息量较大，我需要时间消化——建议大家关注投票链和发言一致性。",
    f"{_FALLBACK_PREFIX}本轮我先听大家的分析，重点观察谁的逻辑链最严密、谁的立场有突变。",
    f"{_FALLBACK_PREFIX}我倾向于保持开放态度，不急于站边——让子弹飞一会儿，看后续发言质量。",
    f"{_FALLBACK_PREFIX}好人阵营需要团结，但也要警惕跟风——独立思考是我给大家的建议。",
)


def context_clues(context: AgentContext) -> str:
    """提取 fallback 理由可复用的公开上下文线索。"""
    clues: list[str] = []
    sheriff_id = context.visible_world_state.get("sheriff_id")
    alive_players = context.visible_world_state.get("alive_players", [])
    if sheriff_id and sheriff_id in alive_players:
        clues.append(f"当前警长是{sheriff_id}")
    for item in context.salience_items[:3]:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or item.get("event")
        if item_type == "seer_claim":
            _append_seer_claim_clue(clues, item)
        elif item_type in {"player_died", "death"}:
            _append_death_clue(clues, item)
        elif item_type == "vote_resolved":
            exiled = item.get("exiled")
            if exiled:
                clues.append(f"上一轮放逐{exiled}")
    if context.recent_transcript:
        last = context.recent_transcript[-1]
        speaker = last.get("speaker")
        text = str(last.get("text") or "").strip()
        if speaker and text:
            clues.append(f"{speaker}最近发言：{text[:24]}")
    return "；".join(clues[:3])


def build_fallback_speech(context: AgentContext) -> str:
    """根据任务类型、身份和公开目标构造兜底发言。"""
    seer_pk = _seer_pk_speech(context)
    if seer_pk:
        return seer_pk

    seed_hash = _fallback_seed_hash(context)
    target = _fallback_speech_target(context, seed_hash)
    tmpl_idx = seed_hash % 7

    if context.task_type == TaskType.WOLF_DISCUSSION:
        return _format_selected(
            _WOLF_DISCUSSION_TARGET_TEMPLATES if target else _WOLF_DISCUSSION_TEMPLATES,
            tmpl_idx,
            target,
        )
    if context.task_type in (TaskType.SHERIFF_SPEECH, TaskType.PK_SPEECH):
        return _SHERIFF_OR_PK_TEMPLATES[tmpl_idx]
    if context.task_type == TaskType.DEFENSE_SPEECH:
        return _DEFENSE_TEMPLATES[tmpl_idx]
    if context.task_type == TaskType.LAST_WORDS:
        return _format_selected(
            _LAST_WORDS_TARGET_TEMPLATES if target else _LAST_WORDS_TEMPLATES,
            tmpl_idx,
            target,
        )
    if target:
        return _DAY_TARGET_TEMPLATES[tmpl_idx].format(target=target)

    logger.warning(
        "fallback speech used for agent=%s day=%s phase=%s task=%s",
        context.agent_id,
        context.day_number,
        context.phase,
        context.task_type,
    )
    return _DAY_TEMPLATES[tmpl_idx]


def _append_seer_claim_clue(clues: list[str], item: dict[str, Any]) -> None:
    speaker = item.get("speaker") or item.get("seer_id")
    target = item.get("target") or item.get("target_id")
    result = item.get("result") or item.get("alignment")
    if speaker and target and result:
        clues.append(f"{speaker}报{target}为{result}")


def _append_death_clue(clues: list[str], item: dict[str, Any]) -> None:
    player = item.get("player_id") or item.get("target_id")
    reason = item.get("reason")
    if player:
        clues.append(f"{player}死亡" + (f"({reason})" if reason else ""))


def _seer_pk_speech(context: AgentContext) -> str:
    # 预言家 PK fallback 必须保留身份标签，避免通用上警模板误导局面。
    if context.own_role != "seer" or context.task_type != TaskType.PK_SPEECH:
        return ""
    check_history = (context.strategy_directive or {}).get("my_check_history", []) or []
    wolf_checks = [
        check
        for check in check_history
        if check.get("alignment") == "wolf" and not check.get("reported")
    ]
    if wolf_checks:
        wolf_check = wolf_checks[0]
        return (
            f"我是预言家，N{wolf_check.get('night', '?')} 验 "
            f"{wolf_check.get('target', '?')} 是狼人。"
            f"我是真的预言家，请跟我投票。"
        )
    return "我是预言家，请给我一次发言机会详细说明查杀。"


def _fallback_seed_hash(context: AgentContext) -> int:
    seed_str = f"{context.agent_id}:{context.day_number}:{context.phase}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


def _fallback_speech_target(context: AgentContext, seed_hash: int) -> str | None:
    legal = list(context.legal_targets) if context.legal_targets else []
    if context.own_role == "werewolf" and context.strategy_directive:
        legal = _remove_wolf_teammates(legal, context.strategy_directive)
    return legal[seed_hash % len(legal)] if legal else None


def _remove_wolf_teammates(
    legal_targets: list[str],
    strategy_directive: dict[str, Any],
) -> list[str]:
    wolf_plan = strategy_directive.get("wolf_team_plan", {})
    teammates = {
        player_id
        for key in ("fake_seer", "pusher", "deep_hook")
        if (player_id := wolf_plan.get(key, "")) in legal_targets
    }
    non_teammates = [target for target in legal_targets if target not in teammates]
    return non_teammates or legal_targets


def _format_selected(
    templates: tuple[str, ...],
    tmpl_idx: int,
    target: str | None,
) -> str:
    text = templates[tmpl_idx]
    return text.format(target=target) if target else text


__all__ = [
    "build_fallback_speech",
    "context_clues",
]
