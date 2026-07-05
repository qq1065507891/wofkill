# -*- coding: utf-8 -*-
"""
功能描述：**：从metrics.py抽离的纯函数，检测action_trace审计字典中的决策合法性与会话信息泄露，供metrics和trace_builder共享避免循环导入
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

# Action types that must carry a non-empty target_id to be legal.
TARGET_REQUIRED_ACTIONS = {
    "vote",
    "wolf_kill",
    "use_poison",
    "check_alignment",
    "choose_master",
    "hunter_shot",
    "badge_transfer",
    "sheriff_vote",
}


def decision_is_legal_from_trace(trace: dict[str, Any]) -> bool | None:
    """Return False when the action violates legal_actions/legal_targets.

    Returns None when legality cannot be decided (no action_type or no
    legal_* lists recorded), so callers can distinguish "unknown" from
    "illegal".
    """
    parsed = trace.get("parsed_action")
    parsed = parsed if isinstance(parsed, dict) else {}
    decision = parsed.get("decision_plan")
    decision = decision if isinstance(decision, dict) else {}
    action_type = str(
        trace.get("final_action_type")
        or parsed.get("action_type")
        or decision.get("action_type")
        or ""
    )
    if not action_type:
        return None
    legal_actions = trace.get("legal_actions")
    if isinstance(legal_actions, list) and legal_actions and action_type not in legal_actions:
        return False
    target_id = parsed.get("target_id") or decision.get("target_id")
    if action_type in TARGET_REQUIRED_ACTIONS and not target_id:
        return False
    legal_targets = trace.get("legal_targets")
    if (
        target_id
        and isinstance(legal_targets, list)
        and legal_targets
        and target_id not in legal_targets
    ):
        return False
    return True


def dialogue_leaked_from_trace(trace: dict[str, Any]) -> bool | None:
    """Return True when the public dialogue text leaks a concealed secret.

    Returns None when there is no dialogue_plan to inspect.
    """
    parsed = trace.get("parsed_action")
    if not isinstance(parsed, dict):
        return None
    dialogue = parsed.get("dialogue_plan")
    if not isinstance(dialogue, dict):
        return None
    public_parts = [
        dialogue.get("public_intent"),
        *(dialogue.get("talking_points") or []),
        parsed.get("reason"),
        parsed.get("speech"),
        parsed.get("speech_text"),
    ]
    public_text = "\n".join(str(part or "") for part in public_parts).lower()
    if not public_text:
        return False
    for secret in dialogue.get("conceal") or []:
        secret_text = str(secret or "").strip().lower()
        if len(secret_text) >= 4 and secret_text in public_text:
            return True
    return any(
        marker in public_text
        for marker in (
            "wolf teammate",
            "my teammate",
            "night kill",
            "private goal",
            "狼队友",
            "我的队友",
            "夜刀",
            "真实身份",
        )
    )
