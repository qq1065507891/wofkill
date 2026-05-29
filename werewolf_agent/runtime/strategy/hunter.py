"""Hunter strategy evaluation functions."""
from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.strategy.seer import public_seer_claimants

logger = logging.getLogger(__name__)


def evaluate_hunter_shot_target(
    gs: GameState,
    hunter_id: str,
    legal_targets: list[str],
    death_reason: str,
) -> dict[str, Any] | None:
    """Score potential shot targets by evidence strength for the hunter."""
    if not legal_targets:
        return None

    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # 公开查杀声明：狼人阵营是最强信号 (+10)
        # 使用 seer_check_claim 公开信息，不直接读取 seer_check 私有事件
        _wolf_check_found = False
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            _ws = build_world_state(gs)
            for f in _ws.facts_of_type("seer_check_claim"):
                val = (f.value or "").lower()
                if f.target_player == pid and ("wolf" in val or "狼" in (f.value or "")):
                    sig.append(f"seer_check_wolf_claim_{f.source_player}")
                    value += 10
                    _wolf_check_found = True
                    break
        except Exception:
            logger.warning("Failed to score wolf-kill target via seer claims", exc_info=True)

        # Counterclaiming seer: high-value target (+6)
        counterclaiming_seers = public_seer_claimants(gs)
        if pid in counterclaiming_seers:
            sig.append("counterclaiming_seer")
            value += 6

        # Publicly accused of being wolf (+4)
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            if speaker == hunter_id or speaker == pid:
                continue
            if pid in text and ("狼" in text or "查杀" in text or "可疑" in text):
                sig.append(f"public_suspect_by_{speaker}")
                value += 4
                break

        # Voted to exile the hunter (+3)
        for e in gs.events:
            if e.type != "vote_resolved":
                continue
            vote_list = e.payload.get("votes") or []
            for vote in vote_list:
                if isinstance(vote, dict) and vote.get("voter") == pid and vote.get("target") == hunter_id:
                    sig.append("voted_exile_hunter")
                    value += 3
                    break

        # Contradiction alerts (+3)
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            from werewolf_agent.cognition.contradiction import ContradictionEngine
            world_state = build_world_state(gs)
            engine = ContradictionEngine()
            alerts = engine.detect(world_state.facts, gs.day_number)
            for alert in alerts:
                if pid in str(alert):
                    sig.append(f"contradiction_{alert.alert_type}")
                    value += 3
                    break
        except Exception:
            logger.warning("Failed to score wolf-kill target via contradiction alerts", exc_info=True)

        # Claimed a power role (+2)
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            claims = e.payload.get("claims") or []
            for claim in claims:
                if claim.get("type") == "role" and claim.get("value") in (
                    "seer", "witch", "hunter",
                ):
                    sig.append(f"claimed_{claim['value']}")
                    value += 2
                    break
            break

        scores[pid] = {"value": value, "signals": sig}

    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    top_value = ranked[0][1]["value"] if ranked else 0
    has_seer_check_wolf = any(
        "seer_check_wolf" in s
        for _, d in ranked
        for s in d["signals"]
    )

    if has_seer_check_wolf:
        advisory = "有明确查杀目标，强烈建议开枪带走该玩家。"
    elif top_value >= 6:
        advisory = "有较高嫌疑目标，建议开枪。"
    elif top_value >= 3:
        advisory = "有一定嫌疑目标，可以开枪，但也可以选择不开枪。"
    else:
        advisory = "无明显高价值目标，建议不开枪（NO_ACTION），避免误伤好人。"

    return {
        "description": "猎人开枪目标价值评估（分数越高越值得开枪）",
        "death_reason": death_reason,
        "ranked_targets": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "recommendation": (
            f"建议开枪带走: {ranked[0][0]}（价值分={ranked[0][1]['value']}，"
            f"信号: {', '.join(ranked[0][1]['signals']) or '无特殊信号'}）"
            if ranked else "无可用开枪目标"
        ),
        "shoot_advisory": advisory,
    }
