"""Standalone output parsing functions extracted from PlayerAgent.

These functions handle JSON repair, extraction, normalization, and
action parsing for player agent LLM output. They are pure functions
(without self) that operate on explicit inputs.
"""

from __future__ import annotations

import json
import logging
import re
from html import unescape
from typing import Any

from pydantic import ValidationError

from werewolf_agent.agents.schemas import (
    ActionType,
    FactionGoal,
    PlayerAction,
    RiskFlag,
    SeerStance,
    VoteBasis,
)

logger = logging.getLogger(__name__)

# Class-level constants needed by standalone functions
CHOICE_TARGET_ACTIONS = {
    ActionType.VOTE,
    ActionType.WOLF_KILL,
    ActionType.USE_POISON,
    ActionType.CHECK_ALIGNMENT,
    ActionType.CHOOSE_MASTER,
    ActionType.HUNTER_SHOT,
    ActionType.BADGE_TRANSFER,
    ActionType.SHERIFF_VOTE,
}

SPEECH_INTENTS = {
    "self_clear": "表水",
    "question_target": "质疑/追问目标",
    "stand_with_seer": "站边预言家或逻辑线",
    "respond_pressure": "回应质疑",
    "push_vote": "提出投票倾向",
    "info_synthesis": "整合多人发言要点，提出综合判断",
    "anti_herd_call": "指出跟票风险，提醒大家独立判断",
}


def repair_json_text(raw: str) -> str:
    """Apply common JSON repairs for LLM output quirks.

    Handles: trailing commas, single-quoted strings, unquoted keys,
    JS-style comments, NaN/Infinity literals, and BOM.
    """
    import re as _re

    text = raw.strip()
    # Remove BOM and zero-width characters
    text = text.replace("﻿", "").replace("​", "")
    # Remove // line comments and /* block comments */
    text = _re.sub(r"//[^\n]*", "", text)
    text = _re.sub(r"/\*.*?\*/", "", text, flags=_re.DOTALL)
    # Replace NaN / Infinity with null
    text = _re.sub(r"\bNaN\b", "null", text)
    text = _re.sub(r"\bInfinity\b|\binf\b", "null", text, flags=_re.IGNORECASE)
    # Fix single-quoted strings → double-quoted (naive but covers common cases)
    # Only outside already-double-quoted strings
    text = _re.sub(r"(?<!\\)'([^']*?)'", r'"\1"', text)
    # Fix unquoted keys: word followed by :
    text = _re.sub(
        r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r'\1"\2":',
        text,
    )
    # Remove trailing commas before } or ]
    text = _re.sub(r",\s*([}\]])", r"\1", text)
    # Collapse multiple consecutive commas
    text = _re.sub(r",\s*,", ",", text)
    return text


def extract_json_object_candidates(text: str) -> list[str]:
    """Extract balanced JSON object candidates from mixed model text."""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:idx + 1])
                start = None

    action_candidates = [
        candidate for candidate in candidates
        if '"action_type"' in candidate or "'action_type'" in candidate
    ]
    return action_candidates or candidates


def extract_parameter_tag_action(text: str) -> dict[str, Any] | None:
    """Extract MiniMax-style <parameter name="...">value</parameter> tool payloads."""
    pairs = re.findall(
        r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</parameter>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not pairs:
        return None

    data: dict[str, Any] = {}
    for key, raw_value in pairs:
        value = unescape(raw_value.strip())
        if value.lower() in {"null", "none"}:
            data[key] = None
        elif key == "confidence":
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
        else:
            data[key] = value
    return data if "action_type" in data else None


def normalize_action_data(data: Any) -> Any:
    """Normalize provider quirks before schema validation."""
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    if isinstance(normalized.get("target_id"), str) and normalized["target_id"].strip().lower() in {
        "",
        "null",
        "none",
    }:
        normalized["target_id"] = None
    if "confidence" in normalized and isinstance(normalized["confidence"], str):
        try:
            normalized["confidence"] = float(normalized["confidence"].strip())
        except ValueError:
            pass
    return normalized


def clean_enum_value(value: Any, allowed: set[str]) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned in allowed else None


def clean_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"未说明", "无", "none", "null"}:
        return ""
    return text


def sanitize_optional_private_fields(data: Any) -> Any:
    """Drop malformed optional audit fields without invalidating core action."""
    if not isinstance(data, dict):
        return data
    private_intent = data.get("private_intent")
    if not isinstance(private_intent, dict):
        return data

    sanitized = dict(data)
    sanitized_intent = dict(private_intent)

    valid_goals = {goal.value for goal in FactionGoal}
    if sanitized_intent.get("faction_goal") not in valid_goals:
        true_role = str(sanitized_intent.get("true_role") or sanitized.get("action_type") or "")
        sanitized_intent["faction_goal"] = (
            FactionGoal.CONFUSE_GOOD.value
            if true_role == "werewolf"
            else FactionGoal.FIND_WOLVES.value
        )

    valid_flags = {flag.value for flag in RiskFlag}
    flags = sanitized_intent.get("risk_flags")
    if isinstance(flags, list):
        sanitized_intent["risk_flags"] = [
            flag for flag in flags
            if isinstance(flag, str) and flag in valid_flags
        ]
    else:
        sanitized_intent["risk_flags"] = []

    sanitized["private_intent"] = sanitized_intent
    return sanitized


def action_from_data(data: Any) -> tuple[PlayerAction | None, str | None]:
    # PlayerAction is a discriminated Union of 16 action-type variants
    # (pipeline-optimization Task 5). ``model_validate`` is overridden on
    # the base class to route the data through the Union's TypeAdapter,
    # which dispatches on the ``action_type`` discriminator.
    data = normalize_action_data(data)
    try:
        return PlayerAction.model_validate(data), None
    except ValidationError as e:
        sanitized = sanitize_optional_private_fields(data)
        if sanitized != data:
            try:
                return PlayerAction.model_validate(sanitized), None
            except ValidationError:
                pass
        return None, f"Schema validation error: {e}"


def parse_action(text: str) -> tuple[PlayerAction | None, str | None]:
    """Parse LLM output into PlayerAction. Returns (action, error)."""
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try direct parse first
    try:
        data = json.loads(cleaned)
        return action_from_data(data)
    except json.JSONDecodeError as direct_error:
        # Repair and retry
        repaired = repair_json_text(cleaned)
        if repaired != cleaned:
            try:
                data = json.loads(repaired)
                return action_from_data(data)
            except json.JSONDecodeError:
                pass  # fall through to extraction

        parameter_data = extract_parameter_tag_action(cleaned)
        if parameter_data is not None:
            action, parse_error = action_from_data(parameter_data)
            if action is not None:
                return action, None
            return None, parse_error

        candidates = extract_json_object_candidates(cleaned)
        if not candidates:
            return None, f"No JSON object found in output"
        first_error: str | None = None
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as e:
                # Try repair on candidate
                repaired = repair_json_text(candidate)
                if repaired != candidate:
                    try:
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        if first_error is None:
                            first_error = f"JSON parse error: {e}"
                        continue
                else:
                    if first_error is None:
                        first_error = f"JSON parse error: {e}"
                    continue
            action, parse_error = action_from_data(data)
            if action is not None:
                return action, None
            if first_error is None:
                first_error = parse_error
        return None, first_error or f"JSON parse error: {direct_error}"


def extract_decision_data(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return normalize_action_data(data), None
        return None, "Decision JSON must be an object"
    except json.JSONDecodeError as direct_error:
        parameter_data = extract_parameter_tag_action(cleaned)
        if parameter_data is not None:
            return normalize_action_data(parameter_data), None
        candidates = extract_json_object_candidates(cleaned)
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return normalize_action_data(data), None
        return None, f"No JSON object found in output: {direct_error}"


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
    return f"{target_id}是当前合法投票候选，需要基于发言、票型和站边继续施压"


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
    target = speech_pressure_target(intent, target_id, context_legal_targets)
    additions: list[str] = []
    if not re.search(r"好人|我是.*?(?:村民|预言家|女巫|猎人|p\d{2})", speech):
        if context_own_role in {"werewolf", "hybrid"}:
            additions.append(f"我是{context_agent_id}视角。")
        else:
            additions.append("我是好人视角。")
    if target and not re.search(r"(?:怀疑\s*p\d{2}|p\d{2}\s*有问题|投\s*p\d{2})", speech):
        additions.append(f"我怀疑{target}有问题。")
    if target and not re.search(r"(?:投|投票|归票|倾向).*?p\d{2}", speech):
        additions.append(f"我倾向投{target}。")
    if not re.search(r"矛盾|前后不一|不合理|查杀|查验|警徽流|对跳|票数|之前说|刚才说", speech):
        additions.append("依据是查验、票型和前后发言矛盾需要继续对上。")
    if additions:
        return speech.rstrip("。") + "。" + "".join(additions)
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


def default_not_voting_reason(legal_targets: list[str], target_id: str) -> str:
    others = [target for target in legal_targets if target != target_id]
    if not others:
        return "本轮只有一个合法投票目标，没有其他可排除候选。"
    return f"暂不投{', '.join(others[:4])}，因为当前可见线索优先指向{target_id}。"


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
    reason = clean_reason(data.get("reason")) or summary
    suspect_reason = clean_reason(data.get("suspect_reason")) or summary
    standing = clean_reason(data.get("standing_with_seer")) or infer_standing_with_seer(salience_items)
    not_voting = clean_reason(data.get("not_voting_reason")) or default_not_voting_reason(
        legal_targets,
        target_id,
    )
    private_reason = clean_reason(data.get("private_reason")) or (
        f"结构化投票修复：在合法候选中选择{target_id}。依据：{reason}"
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


def parse_choice_action(
    text: str,
    legal_actions: list[ActionType],
    legal_targets: list[str],
    salience_items: list[dict[str, Any]],
) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
    data, parse_error = extract_decision_data(text)
    if data is None:
        return None, parse_error, None
    if "choice" not in data and "target_id" not in data:
        return None, "Choice output must include choice or target_id", data

    if legal_actions == [ActionType.VOTE]:
        repaired = repair_vote_decision(data, legal_actions, legal_targets, salience_items)
    else:
        repaired = repair_target_decision(data, legal_actions, legal_targets, salience_items)
    if repaired is None:
        return None, "Could not map choice to legal target", data

    action = PlayerAction(
        action_type=legal_actions[0],
        target_id=repaired["target_id"],
        speech="",
        reason=repaired["reason"],
        confidence=repaired["confidence"],
        seer_stance=repaired.get("seer_stance", SeerStance.UNDECIDED.value),
        vote_basis=repaired.get("vote_basis", VoteBasis.FALLBACK.value),
        standing_with_seer=repaired.get("standing_with_seer", ""),
        suspect_reason=repaired.get("suspect_reason", ""),
        not_voting_reason=repaired.get("not_voting_reason", ""),
        private_reason=repaired.get("private_reason", ""),
    )
    return action, None, repaired


def parse_speech_intent_action(
    text: str,
    context_agent_id: str,
    context_own_role: str | None,
    context_legal_targets: list[str],
    context_salience_items: list[dict[str, Any]],
    context_visible_world_state: dict[str, Any],
    context_recent_transcript: list[dict[str, Any]],
) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
    data, parse_error = extract_decision_data(text)
    if data is None:
        return None, parse_error, None
    if "intent" not in data and "speech" not in data:
        return None, "Speech intent output must include intent or speech", data

    repaired = repair_speech_intent_decision(
        data,
        context_agent_id,
        context_own_role,
        context_legal_targets,
        context_salience_items,
        context_visible_world_state,
        context_recent_transcript,
    )
    action = PlayerAction(
        action_type=ActionType.SPEECH,
        target_id=repaired["target_id"],
        speech=repaired["speech"],
        reason=repaired["reason"],
        confidence=repaired["confidence"],
    )
    return action, None, repaired


def uses_choice_pipeline(legal_actions: list[ActionType], legal_targets: list[str]) -> bool:
    return (
        len(legal_actions) == 1
        and legal_actions[0] in CHOICE_TARGET_ACTIONS
        and bool(legal_targets)
    )


def uses_speech_intent_pipeline(
    legal_actions: list[ActionType],
    task_type: Any,
    speech_intent_tasks: set,
) -> bool:
    return (
        task_type in speech_intent_tasks
        and legal_actions == [ActionType.SPEECH]
    )
