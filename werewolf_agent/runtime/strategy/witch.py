"""Witch strategy evaluation functions."""
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
                "save_now": "【强烈建议使用解药】首夜救人是标准女巫的第一优先级操作。被杀者有{:.0f}%概率是神职（预言家/猎人/白痴/混血儿），若不救则好人立即损失关键角色。除非你有钢铁般的自刀证据（仅在被杀者上警且发言诡异的极罕见情况），否则首夜必须救人。错过首夜救人意味着解药大概率浪费——N2之后被刀的人通常没有遗言，且狼队会优先刀明好人".format(
                    power_roles_alive / max(non_wolf_alive, 1) * 100,
                ),
                "save_later": "【不要保留解药！】解药留到N2+的收益远不如首夜救人——N1死亡有遗言但N2+没有，且真预言家通常在N1就被刀的。保留解药的最大风险是被刀者是预言家，好人直接崩盘",
                "risk_no_save": "【不救的风险极高】不救→被刀者必死→如果他是预言家/猎人，好人阵营胜率直接掉30%以上。只有面对极其明显的自刀（概率<5%）才考虑不救",
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
            + ("高价值目标，强烈建议救人。" if score >= 6 else
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
