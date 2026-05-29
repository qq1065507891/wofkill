"""Seer strategy evaluation functions."""
from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def public_seer_claimants(gs: GameState) -> set[str]:
    """Return public players who have claimed seer in speeches."""
    claimants: set[str] = set()
    seer_markers = (
        "我是预言家",
        "我跳预言家",
        "认预言家",
        "悍跳预言家",
        "claim seer",
        "claimed seer",
        "i am the seer",
    )
    for event in gs.events:
        if event.type not in ("speech", "sheriff_speech"):
            continue
        speaker = event.payload.get("speaker")
        if not speaker:
            continue
        claims = event.payload.get("claims") or []
        for claim in claims:
            if claim.get("type") == "role" and claim.get("value") == "seer":
                claimants.add(speaker)
                break
        else:
            text = str(event.payload.get("text", "")).lower()
            if any(marker in text for marker in seer_markers):
                claimants.add(speaker)
    return claimants


def evaluate_seer_check_value(
    gs: GameState,
    seer_id: str,
    legal_targets: list[str],
) -> dict[str, Any] | None:
    """Score unchecked targets by information value for the seer."""
    if not legal_targets:
        return None

    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # High-value: was accused of being wolf in public speech
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            if speaker == seer_id:
                continue
            if pid in text and ("狼" in text or "可疑" in text or "查杀" in text):
                sig.append(f"public_suspect_by_{speaker}")
                value += 3
                break

        # High-value: claimed a power role — verify authenticity
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            claims = e.payload.get("claims") or []
            for claim in claims:
                if claim.get("type") == "role" and claim.get("value") in (
                    "seer", "witch", "hunter",
                ):
                    sig.append(f"claimed_{claim['value']}")
                    value += 5
                    break
            if "女巫" in text or "我是猎人" in text:
                sig.append("claimed_power_in_text")
                value += 4
            break

        # Medium: ran for sheriff (potential power role or wolf)
        for e in gs.events:
            if e.type == "sheriff_registration" and e.payload.get("player_id") == pid:
                sig.append("ran_for_sheriff")
                value += 2
                break

        # Medium: unclear stance — not clearly aligned with any seer claimant
        seer_claimants = public_seer_claimants(gs)
        if seer_claimants:
            supported_a_seer = False
            for e in gs.events:
                if e.type not in ("speech", "sheriff_speech"):
                    continue
                if e.payload.get("speaker") != pid:
                    continue
                text = str(e.payload.get("text", ""))
                if any(sc in text for sc in seer_claimants):
                    supported_a_seer = True
                    break
            if not supported_a_seer:
                sig.append("unclear_stance")
                value += 3

        # Low: active speaker (more data available for LLM to judge)
        speech_count = sum(
            1 for e in gs.events
            if e.type in ("speech", "sheriff_speech")
            and e.payload.get("speaker") == pid
        )
        if speech_count == 0:
            sig.append("silent_player")
            value += 2

        scores[pid] = {"value": value, "signals": sig}

    # Sort by value descending
    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    return {
        "description": "未验玩家的信息价值评估（分数越高越值得验）",
        "ranked_targets": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "recommendation": (
            f"建议优先查验: {ranked[0][0]}（价值分={ranked[0][1]['value']}，"
            f"信号: {', '.join(ranked[0][1]['signals']) or '无特殊信号'}）"
            if ranked else "无可用验人目标"
        ),
    }
