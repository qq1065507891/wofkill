# -*- coding: utf-8 -*-
"""
处理 LLM 原始输出到 PlayerAction 的解析、归一化和修复链路。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.agents.output_parser import parse_action
    >>> parse_action(raw_text)
"""

from __future__ import annotations

import json
import logging
import re
from html import unescape
from typing import Any

from pydantic import ValidationError

from werewolf_agent.agents.json_repair import (
    extract_json_object_candidates,
    repair_json_text,
)
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


def extract_partial_decision_data(text: str) -> dict[str, Any] | None:
    """Recover discriminator fields from a truncated JSON object.

    This is intentionally narrow: only short completed scalar fields that
    identify a target-choice decision are recovered. Long free-form fields
    such as speech/reason are left to the existing repair pipeline to
    synthesize from the chosen legal target.
    """
    if "{" not in text:
        return None
    data: dict[str, Any] = {}
    for key in ("choice", "action_type", "target_id"):
        match = re.search(rf'"{key}"\s*:\s*"([^"\\]*)"', text)
        if match:
            data[key] = match.group(1)
    confidence = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if confidence:
        try:
            data["confidence"] = float(confidence.group(1))
        except ValueError:
            pass
    if "choice" in data or "target_id" in data:
        return normalize_action_data(data)
    return None


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


# P1-G3223805846-5: 常见 LLM 字段名 typo 归一化映射。LLM 经常写出
# 拼写错误的字段名（如 `not_vading_reason`、`targe_id`），直接在解析
# 入口做归一化，避免下游 Pydantic 校验把这些数据当 schema error
# 丢弃。映射表只覆盖反复出现过的 typo，不要试图做成模糊匹配。
_TYPO_ALIASES: dict[str, str] = {
    "not_vading_reason": "not_voting_reason",
    "not_vote_reason": "not_voting_reason",
    "targe_id": "target_id",
    "targt_id": "target_id",
}


def _normalize_typos(data: dict[str, Any]) -> dict[str, Any]:
    """P1-G3223805846-5: 归一常见 LLM typo。返回新 dict，不修改原对象。

    仅在原 dict 同时缺少正确字段名时才替换，避免覆盖下游已经
    填好的合法值。返回新对象，调用方拿到 dict 即可安全复用。
    """
    if not isinstance(data, dict):
        return data
    result = dict(data)
    for typo, correct in _TYPO_ALIASES.items():
        if typo in result and correct not in result:
            result[correct] = result.pop(typo)
    return result


def clean_enum_value(value: Any, allowed: set[str]) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned in allowed else None


# D4-6 (P2): placeholder filter set for `clean_reason`. The previous
# hard-coded set was only 4 entries; real LLM output produces ~15
# distinct placeholder strings for the reason / private_reason /
# suspect_reason fields. Anything that survives the parser gets
# logged into the audit trail and surfaced in the dashboard,
# polluting downstream review. Set is split 8 Chinese + 4 English +
# 3 punctuation to make the regression test parametrization
# explicit.
_REASON_PLACEHOLDERS: frozenset[str] = frozenset({
    # Chinese (8)
    "未说明",
    "无",
    "未知",
    "不清楚",
    "暂无",
    "未填",
    "无理由",
    "没办法",
    # English (4)
    "none",
    "null",
    "N/A",
    "n/a",
    # Punctuation (3) — what happens when the LLM gives up mid-thought
    "-",
    "?",
    "...",
})


def clean_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in _REASON_PLACEHOLDERS:
        # D4-6 (P2): surface the placeholder substitution to ops. A
        # silent filter would lose the signal that the LLM is
        # filling the reason field with garbage. We only log on
        # the filter path (not the empty-string path) so the
        # volume stays proportional to the LLM's actual failure
        # rate.
        if text and text in _REASON_PLACEHOLDERS:
            logger.warning(
                "clean_reason: filtered placeholder reason %r to ''",
                text,
            )
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

    # P1-S7 (residual): claimed_view is documented as an identity-
    # perspective identifier (PrivateIntent schema), not a free-form
    # Chinese phrase. Game trace g_3528592081 showed real wolves
    # writing "我是好人，混水摸鱼" — a strategy note in natural
    # Chinese. Sanitize any non-enum value to a safe default so the
    # audit log / dashboard only sees clean identifiers. The valid set
    # is the union of: the canonical safe default, all role names, and
    # the seer-specific "seer" identifier. Anything else gets replaced.
    raw_claimed = sanitized_intent.get("claimed_view")
    if not isinstance(raw_claimed, str) or raw_claimed not in _VALID_CLAIMED_VIEW_VALUES:
        # Detect the LLM writing a Chinese natural-language claim —
        # if it contains Chinese characters and isn't in the valid set,
        # it's almost certainly the bad pattern from the game trace.
        sanitized_intent["claimed_view"] = _safe_default_claimed_view(
            sanitized_intent.get("true_role"),
        )

    sanitized["private_intent"] = sanitized_intent
    return sanitized


# P1-S7 (residual): enum-like identifiers acceptable as claimed_view.
# The safe default "good_player_without_night_info" is the standard
# good-side claim; role names are valid because a wolf can claim any
# role publicly (e.g., "villager", "witch"). "seer" is canonical for
# the seer's public claim; "good_player_without_night_info" is the
# generic catch-all.
_VALID_CLAIMED_VIEW_VALUES: frozenset[str] = frozenset({
    "good_player_without_night_info",
    "seer",
    "werewolf",
    "villager",
    "witch",
    "hunter",
    "idiot",
    "hybrid",
})


def _safe_default_claimed_view(true_role: Any) -> str:
    """Pick a safe default claimed_view based on the agent's true_role.

    - seer → "seer" (the only public claim that makes sense for seer)
    - everything else → "good_player_without_night_info" (the standard
      good-side cover, used by all non-wolf roles and by wolves
      pretending to be good)
    """
    if isinstance(true_role, str) and true_role == "seer":
        return "seer"
    return "good_player_without_night_info"


def action_from_data(data: Any) -> tuple[PlayerAction | None, str | None]:
    # PlayerAction is a discriminated Union of 16 action-type variants
    # (pipeline-optimization Task 5). ``model_validate`` is overridden on
    # the base class to route the data through the Union's TypeAdapter,
    # which dispatches on the ``action_type`` discriminator.
    if isinstance(data, dict):
        # P1-G3223805846-5: 解析入口归一常见 LLM 字段名 typo
        # (如 `not_vading_reason` → `not_voting_reason`)。
        data = _normalize_typos(data)
    data = normalize_action_data(data)
    # P1-S7 (residual): always sanitize private_intent before validation.
    # The previous code only sanitized on validation failure, which let
    # free-form Chinese claimed_view strings (e.g., "我是好人，混水摸鱼")
    # pass through cleanly. Sanitizing first normalizes claimed_view to
    # an enum-style identifier so the audit log / dashboard see only
    # clean values.
    data = sanitize_optional_private_fields(data)
    try:
        return PlayerAction.model_validate(data), None
    except ValidationError as e:
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
        partial_data = extract_partial_decision_data(cleaned)
        if partial_data is not None:
            return partial_data, None
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


def _target_choice_action(legal_actions: list[ActionType]) -> ActionType | None:
    for action in legal_actions:
        if action in CHOICE_TARGET_ACTIONS:
            return action
    return None


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

    target_action = _target_choice_action(legal_actions)
    if target_action is None:
        return None, "No target-required action available for choice output", data

    if target_action == ActionType.VOTE:
        repaired = repair_vote_decision(data, legal_actions, legal_targets, salience_items)
    else:
        repaired = repair_target_decision(data, legal_actions, legal_targets, salience_items)
    if repaired is None:
        return None, "Could not map choice to legal target", data

    # P0-S8: only VOTE actions carry vote-audit fields. With
    # ``extra="forbid"`` on every variant, passing them to e.g.
    # WolfKillPlayerAction is now a parse error — so we only attach
    # them when the action is actually a vote.
    if target_action == ActionType.VOTE:
        action = PlayerAction(
            action_type=target_action,
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
    else:
        action = PlayerAction(
            action_type=target_action,
            target_id=repaired["target_id"],
            speech="",
            reason=repaired["reason"],
            confidence=repaired["confidence"],
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
        intent=repaired["intent"],
    )
    return action, None, repaired


def uses_choice_pipeline(legal_actions: list[ActionType], legal_targets: list[str]) -> bool:
    target_actions = [action for action in legal_actions if action in CHOICE_TARGET_ACTIONS]
    non_target_actions = [action for action in legal_actions if action not in CHOICE_TARGET_ACTIONS]
    if len(legal_actions) == 1:
        return legal_actions[0] in CHOICE_TARGET_ACTIONS and bool(legal_targets)
    return (
        len(target_actions) == 1
        and target_actions[0] == ActionType.SHERIFF_VOTE
        and set(non_target_actions).issubset({ActionType.NO_ACTION})
        and bool(legal_targets)
    )


def uses_speech_intent_pipeline(
    legal_actions: list[ActionType],
    task_type: Any,
    speech_intent_tasks: set,
) -> bool:
    speech_only = legal_actions == [ActionType.SPEECH]
    legacy_speech_vote = set(legal_actions) == {ActionType.SPEECH, ActionType.VOTE}
    return (
        task_type in speech_intent_tasks
        and (speech_only or legacy_speech_vote)
    )
