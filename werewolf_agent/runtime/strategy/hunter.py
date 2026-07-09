# -*- coding: utf-8 -*-
"""
功能描述：猎人策略评估函数。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-09
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.strategy._shared import (
    speech_is_negated as _speech_is_negated,
)
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

    public_good_claim_sources = _public_good_claim_sources(gs, legal_targets)
    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # 公开金水/认好声明降低开枪价值；只使用公开信息，不读取隐藏身份。
        for source in sorted(public_good_claim_sources.get(pid, set())):
            sig.append(f"public_good_claim_by_{source}")
            value -= 6

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
            # D-6: skip negated power-role claims.  A speech that
            # says "我不是女巫" or "否认我是预言家" must not score
            # as a positive claim.
            text = str(e.payload.get("text", ""))
            if _speech_is_negated(text):
                continue
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


def _public_good_claim_sources(
    gs: GameState,
    legal_targets: list[str],
) -> dict[str, set[str]]:
    """返回公开认好或给出金水声明的来源玩家。"""
    legal = set(legal_targets)
    sources: dict[str, set[str]] = {pid: set() for pid in legal_targets}

    try:
        from werewolf_agent.cognition.public_evidence import is_good_result
        from werewolf_agent.cognition.world_state import build_world_state

        ws = build_world_state(gs)
        for fact in ws.facts_of_type("seer_check_claim"):
            target = fact.target_player
            source = fact.source_player
            if (
                target in legal
                and source
                and source != target
                and is_good_result(str(fact.value or ""))
            ):
                sources.setdefault(target, set()).add(source)
        for fact in ws.facts_of_type("claimed_good"):
            target = fact.target_player
            source = fact.source_player
            if target in legal and source and source != target:
                sources.setdefault(target, set()).add(source)
    except Exception:
        logger.warning("Failed to collect public good claim sources", exc_info=True)

    for event in gs.events:
        if event.type not in ("speech", "sheriff_speech"):
            continue
        text = str(event.payload.get("text", ""))
        speaker = event.payload.get("speaker")
        if not text or not speaker:
            continue
        for target in legal:
            if speaker == target or target not in text:
                continue
            if _negates_good_claim(text, target):
                continue
            if "金水" in text or "好人" in text or f"保{target}" in text:
                sources.setdefault(target, set()).add(str(speaker))
    return sources


def _negates_good_claim(text: str, target: str) -> bool:
    negated_patterns = (
        f"{target}不是好人",
        f"{target}不是金水",
        f"不保{target}",
        f"{target}不保",
    )
    return any(pattern in text for pattern in negated_patterns)
