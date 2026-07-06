# -*- coding: utf-8 -*-
"""
修复 target-choice 与投票 decision，使其匹配合法动作和候选目标。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.action_repair import vote_choice_map
    >>> vote_choice_map(["p01", "p02"])
"""

from __future__ import annotations

import json
import re
from typing import Any

from werewolf_agent.agents.action_normalization import clean_enum_value, clean_reason
from werewolf_agent.agents.schemas import ActionType, SeerStance, VoteBasis
from werewolf_agent.agents.speech_intent_parser import (
    SPEECH_INTENTS,
    ensure_speech_quality_components,
    infer_seer_stance,
    infer_speech_intent,
    infer_standing_with_seer,
    infer_vote_basis,
    speech_intent_reason,
    speech_target_from_decision,
    synthesize_intent_speech,
)


def vote_choice_map(legal_targets: list[str]) -> dict[str, str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {
        letters[idx]: target
        for idx, target in enumerate(legal_targets[:len(letters)])
    }


def target_from_vote_decision(
    data: dict[str, Any],
    choice_map: dict[str, str],
    legal_targets: list[str],
) -> str | None:
    choice = str(data.get("choice") or "").strip().upper()
    if choice in choice_map:
        return choice_map[choice]
    target = data.get("target_id")
    if isinstance(target, str) and target in legal_targets:
        return target
    haystack = " ".join(str(value) for value in data.values())
    for candidate in legal_targets:
        if candidate in haystack:
            return candidate
    if len(legal_targets) == 1:
        return legal_targets[0]
    return None


def choice_for_target(choice_map: dict[str, str], target_id: str) -> str:
    for choice, mapped_target in choice_map.items():
        if mapped_target == target_id:
            return choice
    return ""


def vote_candidate_summary(
    salience_items: list[dict[str, Any]],
    target_id: str,
) -> str:
    clues: list[str] = []
    for item in salience_items:
        if not isinstance(item, dict):
            continue
        target = item.get("target") or item.get("target_id") or item.get("player_id")
        if target != target_id:
            continue
        item_type = item.get("type") or item.get("event")
        if item_type == "seer_claim":
            speaker = item.get("speaker") or item.get("seer_id")
            result = item.get("result") or item.get("alignment")
            if speaker and result:
                clues.append(f"{speaker}报{target_id}为{result}")
        elif item_type in {"vote_resolved", "vote"}:
            clues.append(f"{target_id}出现在关键票型中")
        elif item_type in {"player_died", "death"}:
            clues.append(f"{target_id}关联死亡事件")
    if clues:
        return "；".join(clues[:2])
    return f"{target_id}当前缺少可引用的具体公开证据，需补充发言、票型或站边中的具体疑点"


def target_candidate_summary(
    legal_actions: list[ActionType],
    salience_items: list[dict[str, Any]],
    target_id: str,
) -> str:
    action = legal_actions[0] if legal_actions else ActionType.NO_ACTION
    action_reasons = {
        ActionType.WOLF_KILL: "作为狼队夜间击杀目标",
        ActionType.USE_POISON: "作为女巫毒药目标",
        ActionType.CHECK_ALIGNMENT: "作为预言家查验目标",
        ActionType.CHOOSE_MASTER: "作为混血儿主人选择目标",
        ActionType.HUNTER_SHOT: "作为猎人开枪目标",
        ActionType.BADGE_TRANSFER: "作为警徽移交目标",
        ActionType.SHERIFF_VOTE: "作为警长投票目标",
    }
    clues: list[str] = []
    for item in salience_items:
        if not isinstance(item, dict):
            continue
        item_text = json.dumps(item, ensure_ascii=False)
        if target_id in item_text:
            clues.append(item_text[:80])
    basis = f"；依据：{'；'.join(clues[:2])}" if clues else ""
    return f"{target_id}{action_reasons.get(action, '作为当前合法目标')}较合适{basis}"


def default_not_voting_reason(legal_targets: list[str], target_id: str) -> str:
    others = [target for target in legal_targets if target != target_id]
    if not others:
        return "本轮只有一个合法投票目标，没有其他可排除候选。"
    return f"暂不投{', '.join(others[:4])}，因为当前可见线索优先指向{target_id}。"


def target_consistent_reason(text: str, target_id: str, fallback: str) -> str:
    mentioned = set(re.findall(r"p\d{2}", text))
    if mentioned and target_id not in mentioned:
        return fallback
    return text


def repair_vote_decision(
    data: dict[str, Any],
    legal_actions: list[ActionType],
    legal_targets: list[str],
    salience_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    choice_map_val = vote_choice_map(legal_targets)
    target_id = target_from_vote_decision(data, choice_map_val, legal_targets)
    if target_id is None:
        return None

    summary = vote_candidate_summary(salience_items, target_id)
    reason = target_consistent_reason(
        clean_reason(data.get("reason")) or summary,
        target_id,
        summary,
    )
    suspect_reason = target_consistent_reason(
        clean_reason(data.get("suspect_reason")) or summary,
        target_id,
        summary,
    )
    standing = clean_reason(data.get("standing_with_seer")) or infer_standing_with_seer(salience_items)
    not_voting = clean_reason(data.get("not_voting_reason")) or default_not_voting_reason(
        legal_targets,
        target_id,
    )
    private_fallback = f"结构化投票修复：在合法候选中选择{target_id}。依据：{reason}"
    private_reason = target_consistent_reason(
        clean_reason(data.get("private_reason")) or private_fallback,
        target_id,
        private_fallback,
    )
    vote_basis = clean_enum_value(
        data.get("vote_basis"),
        {basis.value for basis in VoteBasis},
    )
    if vote_basis is None:
        vote_basis = infer_vote_basis(reason, suspect_reason, private_reason)
    seer_stance = clean_enum_value(
        data.get("seer_stance"),
        {stance.value for stance in SeerStance},
    )
    if seer_stance is None:
        seer_stance = infer_seer_stance(salience_items, standing)
    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "choice": choice_for_target(choice_map_val, target_id),
        "target_id": target_id,
        "reason": reason,
        "seer_stance": seer_stance,
        "vote_basis": vote_basis,
        "standing_with_seer": standing,
        "suspect_reason": suspect_reason,
        "not_voting_reason": not_voting,
        "private_reason": private_reason,
        "confidence": confidence,
    }


def repair_target_decision(
    data: dict[str, Any],
    legal_actions: list[ActionType],
    legal_targets: list[str],
    salience_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    choice_map_val = vote_choice_map(legal_targets)
    target_id = target_from_vote_decision(data, choice_map_val, legal_targets)
    if target_id is None:
        return None

    reason = clean_reason(data.get("reason")) or target_candidate_summary(
        legal_actions,
        salience_items,
        target_id,
    )
    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "choice": choice_for_target(choice_map_val, target_id),
        "target_id": target_id,
        "reason": reason,
        "confidence": confidence,
    }


def repair_speech_intent_decision(
    data: dict[str, Any],
    context_agent_id: str,
    context_own_role: str | None,
    context_legal_targets: list[str],
    context_salience_items: list[dict[str, Any]],
    context_visible_world_state: dict[str, Any],
    context_recent_transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    intent = str(data.get("intent") or "").strip()
    if intent not in SPEECH_INTENTS:
        intent = infer_speech_intent(data, context_legal_targets)
    target_id = speech_target_from_decision(data, context_legal_targets)
    speech = clean_reason(data.get("speech"))
    reason = clean_reason(data.get("reason"))
    if not speech:
        speech = synthesize_intent_speech(
            intent, target_id,
            context_salience_items, context_visible_world_state,
            context_recent_transcript, context_legal_targets,
        )
    speech = ensure_speech_quality_components(
        speech, intent, target_id,
        context_own_role, context_agent_id, context_legal_targets,
    )
    if not reason:
        reason = speech_intent_reason(intent, target_id)
    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return {
        "intent": intent,
        "target_id": target_id,
        "speech": speech,
        "reason": reason,
        "confidence": confidence,
    }
