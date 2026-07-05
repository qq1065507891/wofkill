# -*- coding: utf-8 -*-
"""
功能描述：女巫策略评估函数。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

import re as _re
from typing import Any

from werewolf_agent.core.models import GameState


def estimate_witch_save_value(
    gs: GameState,
    target_id: str | None,
) -> dict[str, Any]:
    """Estimate the strategic value of saving the wolf-kill target.

    Returns structured decision data — the LLM still makes the final call.
    On N1 there is no public info, so we provide a probability framework
    and explicit trade-offs instead of a numeric score.
    On N2+ we score the target based on observable behavior.
    """
    if target_id is None:
        return {"actionable": False, "reason": "no_wolf_kill"}

    non_wolf_alive = sum(
        1 for p in gs.players.values() if p.alive and p.role != "werewolf"
    )
    power_roles_alive = sum(
        1 for p in gs.players.values()
        if p.alive and p.role in ("seer", "hunter", "idiot", "hybrid")
    )

    # N1: no public information, use probability framework
    if gs.night_number == 1 and gs.day_number == 0:
        return {
            "actionable": True,
            "night": 1,
            "public_info_available": False,
            "probability_framework": {
                "p_seer": round(1 / max(non_wolf_alive, 1), 3),
                "p_power_role": round(power_roles_alive / max(non_wolf_alive, 1), 3),
                "p_villager": round(
                    max(non_wolf_alive - power_roles_alive, 0) / max(non_wolf_alive, 1), 3
                ),
            },
            "trade_off": {
                "save_now": "立即使用解药可以保住当前刀口；按存活构成估算，被杀者约有{:.0f}%概率是神职。首夜公开信息少，这一概率只能作为收益参考，不能替代最终判断。".format(
                    power_roles_alive / max(non_wolf_alive, 1) * 100,
                ),
                "save_later": "保留解药可以等待后续更明确的身份与票型信息，但承担当前刀口直接死亡、药水未来可能来不及使用的风险。",
                "risk_no_save": "不救的主要风险是当前目标可能为关键好人；救人的主要风险是目标价值不足或公开信息误导。请比较两侧机会成本。",
            },
        }

    # N2+: score target based on observable public behavior
    score = 0
    signals: list[str] = []

    # Was target the sheriff or sheriff candidate?
    if gs.sheriff_id == target_id and gs.sheriff_badge_state == "active":
        score += 8
        signals.append("is_sheriff")
    for e in gs.events:
        if e.type == "sheriff_registration" and e.payload.get("player_id") == target_id:
            score += 3
            signals.append("ran_for_sheriff")
            break

    # Did target claim seer or power role in public speech?
    for e in gs.events:
        if e.type != "speech" or e.payload.get("speaker") != target_id:
            continue
        text = e.payload.get("text", "")
        if "预言家" in text or "seer" in text.lower():
            score += 6
            signals.append("claimed_seer_in_speech")
            break
        if "猎人" in text or "白痴" in text:
            score += 2
            signals.append("claimed_power_role")

    # Did anyone else confirm target as good (seer check result)?
    for e in gs.events:
        if e.type == "speech":
            text = e.payload.get("text", "")
            if target_id in text and ("金水" in text or "好人" in text):
                score += 3
                signals.append("confirmed_good_by_seer_claim")
                break

    # How many speeches did target give? (active participants are usually power roles)
    speech_count = sum(
        1 for e in gs.events
        if e.type == "speech" and e.payload.get("speaker") == target_id
    )
    if speech_count >= 2:
        score += 1
        signals.append(f"active_speaker({speech_count}_speeches)")

    return {
        "actionable": True,
        "night": gs.night_number,
        "public_info_available": True,
        "target_id": target_id,
        "save_value_score": score,
        "signals": signals,
        "interpretation": (
            f"目标公开行为分析：得分{score}分。"
            + ("高价值目标，证据倾向救人。" if score >= 6 else
              "中等价值目标，需综合判断。" if score >= 3 else
              "低价值目标，可考虑保留解药。")
        ),
    }


def build_witch_pressure_targets(gs: GameState) -> list[dict[str, Any]]:
    """Build poison pressure targets from public state.

    Pressure sources:
    - Unresolved seer black claim (查杀 in public speech)
    - Player contradicted claimed role
    """
    targets: dict[str, dict[str, Any]] = {}

    # Extract from public speeches: black claims (查杀)
    for event in gs.events:
        if event.type == "speech":
            text = event.payload.get("text", "")
            # Look for 查杀 claims targeting a player (support various ID formats)
            # Pattern: "PLAYER查杀" or "PLAYER...查杀"
            black_match = _re.search(r"([a-zA-Z]+\d+).*?查杀", text)
            if not black_match:
                # Also try reverse: "查杀...PLAYER"
                black_match = _re.search(r"查杀.*?([a-zA-Z]+\d+)", text)
            if black_match:
                target_id = black_match.group(1)
                if target_id not in targets:
                    targets[target_id] = {
                        "player_id": target_id,
                        "pressure_type": "black_claim",
                        "source": event.payload.get("speaker", ""),
                        "description": f"被{event.payload.get('speaker', '?')}查杀",
                    }

    return list(targets.values())
