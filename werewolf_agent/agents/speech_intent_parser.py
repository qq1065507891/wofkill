# -*- coding: utf-8 -*-
"""
推断公开发言意图并合成兜底发言文本。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.speech_intent_parser import infer_speech_intent
    >>> infer_speech_intent({"speech": "我想追问p02"}, ["p02"])
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import SeerStance, VoteBasis

logger = logging.getLogger(__name__)

SPEECH_INTENTS = {
    "self_clear": "表水",
    "question_target": "质疑/追问目标",
    "stand_with_seer": "站边预言家或逻辑线",
    "respond_pressure": "回应质疑",
    "push_vote": "提出投票倾向",
    "info_synthesis": "整合多人发言要点，提出综合判断",
    "anti_herd_call": "指出跟票风险，提醒大家独立判断",
}


def infer_speech_intent(data: dict[str, Any], legal_targets: list[str]) -> str:
    text = " ".join(str(value) for value in data.values())
    if any(word in text for word in ("站边", "预言家", "查验")):
        return "stand_with_seer"
    if any(word in text for word in ("整合", "综合", "多人", "总结")):
        return "info_synthesis"
    if any(word in text for word in ("跟票", "抱团", "独立判断")):
        return "anti_herd_call"
    if any(word in text for word in ("投", "归票", "出")):
        return "push_vote"
    if any(word in text for word in ("回应", "解释", "表水")):
        return "respond_pressure"
    if legal_targets:
        return "question_target"
    return "self_clear"


def speech_target_from_decision(
    data: dict[str, Any],
    legal_targets: list[str],
) -> str | None:
    target = data.get("target_id")
    if isinstance(target, str) and target in legal_targets:
        return target
    haystack = " ".join(str(value) for value in data.values())
    for candidate in legal_targets:
        if candidate in haystack:
            return candidate
    return legal_targets[0] if len(legal_targets) == 1 else None


def _context_clues(
    salience_items: list[dict[str, Any]],
    visible_world_state: dict[str, Any],
    recent_transcript: list[dict[str, Any]],
    agent_id: str = "",
) -> str:
    """Extract context clues for speech synthesis and fallback reasons."""
    clues: list[str] = []
    sheriff_id = visible_world_state.get("sheriff_id")
    alive_players = visible_world_state.get("alive_players", [])
    if sheriff_id and sheriff_id in alive_players:
        clues.append(f"当前警长是{sheriff_id}")
    for item in salience_items[:3]:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or item.get("event")
        if item_type == "seer_claim":
            speaker = item.get("speaker") or item.get("seer_id")
            target = item.get("target") or item.get("target_id")
            result = item.get("result") or item.get("alignment")
            if speaker and target and result:
                clues.append(f"{speaker}报{target}为{result}")
        elif item_type in {"player_died", "death"}:
            player = item.get("player_id") or item.get("target_id")
            reason = item.get("reason")
            if player:
                clues.append(f"{player}死亡" + (f"({reason})" if reason else ""))
        elif item_type == "vote_resolved":
            exiled = item.get("exiled")
            if exiled:
                clues.append(f"上一轮放逐{exiled}")
    if recent_transcript:
        last = recent_transcript[-1]
        speaker = last.get("speaker")
        text = str(last.get("text") or "").strip()
        if speaker and text:
            clues.append(f"{speaker}最近发言：{text[:24]}")
    return "；".join(clues[:3])


def synthesize_intent_speech(
    intent: str,
    target_id: str | None,
    context_salience_items: list[dict[str, Any]],
    context_visible_world_state: dict[str, Any],
    context_recent_transcript: list[dict[str, Any]],
    context_legal_targets: list[str],
) -> str:
    target = target_id or (context_legal_targets[0] if context_legal_targets else "")
    clue = _context_clues(context_salience_items, context_visible_world_state, context_recent_transcript)
    basis = f"结合{clue}，" if clue else ""
    if intent == "stand_with_seer":
        return (
            f"{basis}我现在需要明确站边和逻辑线。"
            f"{('我更倾向相信' + target + '这边的信息，') if target else ''}"
            "接下来会继续核对查验、票型和发言是否能互相印证。"
        )
    if intent == "respond_pressure":
        return (
            f"{basis}我先回应当前压力：我的判断不是跟票，"
            "而是基于发言前后、票型变化和关键事件来排顺序。"
        )
    if intent == "push_vote":
        if target:
            return (
                f"{basis}我这轮会把投票压力先放到{target}。"
                f"{target}需要解释自己的站边、票型和对关键事件的回应。"
            )
        return f"{basis}我这轮会明确给出投票倾向，不接受继续模糊站边。"
    if intent == "info_synthesis":
        return (
            f"{basis}我先把多人发言合在一起看：查验、死亡信息、票型和发言矛盾要能互相印证。"
            "如果只剩单点发言可疑，我不会直接把它放到最高优先级。"
        )
    if intent == "anti_herd_call":
        return (
            f"{basis}我提醒大家不要无条件跟票。"
            "如果同一批人持续集中冲同一个位置，要反查他们有没有抱团带节奏的可能。"
        )
    if intent == "question_target" and target:
        return (
            f"{basis}我想追问{target}：你的站边、票型和关键发言需要正面解释。"
            "如果解释仍然空泛，我会继续把你放在重点怀疑位。"
        )
    return (
        f"{basis}我先把自己的视角说清楚：我会按查验、死亡、票型和发言一致性来判断，"
        "不会只跟随场上声音。"
    )


def ensure_speech_quality_components(
    speech: str,
    intent: str,
    target_id: str | None,
    context_own_role: str | None,
    context_agent_id: str,
    context_legal_targets: list[str],
) -> str:
    # Public speech is semantic output, not a partially filled template.
    # Structural repair must never invent a stance, suspicion, or vote that
    # the model did not express.
    return speech


def speech_pressure_target(
    intent: str,
    target_id: str | None,
    context_legal_targets: list[str],
) -> str | None:
    if intent == "stand_with_seer" and target_id:
        for candidate in context_legal_targets:
            if candidate != target_id:
                return candidate
    return target_id or (context_legal_targets[0] if context_legal_targets else None)


def speech_intent_reason(intent: str, target_id: str | None) -> str:
    intent_label = SPEECH_INTENTS.get(intent, "补充发言")
    if target_id:
        return f"按发言意图「{intent_label}」围绕{target_id}组织公开发言"
    return f"按发言意图「{intent_label}」组织公开发言"


def infer_standing_with_seer(salience_items: list[dict[str, Any]]) -> str:
    for item in salience_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or item.get("event")
        speaker = item.get("speaker") or item.get("seer_id")
        if item_type == "seer_claim" and speaker:
            return str(speaker)
    return ""


def infer_seer_stance(salience_items: list[dict[str, Any]], standing_with_seer: str) -> str:
    if standing_with_seer:
        return SeerStance.TRUST.value
    for item in salience_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or item.get("event")
        if item_type == "seer_claim":
            return SeerStance.UNDECIDED.value
    return SeerStance.NO_CLAIM.value


def infer_vote_basis(*texts: str) -> str:
    try:
        from werewolf_agent.runtime.vote_quality import (
            extract_vote_basis,
            normalize_vote_basis,
        )

        detected = extract_vote_basis(" ".join(text for text in texts if text))
        return normalize_vote_basis(detected)
    except Exception:
        logger.debug("Vote basis inference failed", exc_info=True)
        return VoteBasis.FALLBACK.value
