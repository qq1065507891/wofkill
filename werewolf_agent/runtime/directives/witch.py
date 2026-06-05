"""Witch day-speech directive builder.

D-1: extracted from agent_adapter's hard-coded ``witch_speech_constraint``
block.  D-7: now consults ``evaluate_death_cause_claims`` so the witch's
day-speech guidance is informed by her private knowledge of what poison /
antidote actions she has taken this game.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_witch_directive(
    gs: GameState | None,
    witch_id: str,
) -> dict[str, Any]:
    """Build the day-speech directive for the witch.

    Returns a dict with at least ``witch_speech_directive`` (the
    structured strategy text shown to the LLM).  When ``gs`` is provided,
    the directive is enriched with the witch's private view of
    death-cause claims via :func:`evaluate_death_cause_claims`.
    """
    parts: dict[str, Any] = {}

    base_text = (
        "你是女巫，你掌握的夜间信息（谁被刀、药水使用情况、救了谁、毒了谁）"
        "是你的核心优势。不要轻易暴露这些信息——一旦公开，狼人会知道你的"
        "药水状态并针对性调整策略。但在以下情况可以适度透露：\n"
        "1) 你即将死亡需要传递关键信息；\n"
        "2) 场上好人阵营信息严重不足，需要你站出来带队；\n"
        "3) 有人假冒女巫需要你自证身份。\n"
        "透露时也要衡量利弊，不要在第一天就全部交底。"
    )
    parts["witch_speech_directive"] = base_text

    if gs is None:
        return parts

    # D-7: enrich with private-knowledge death-cause evaluation so the
    # witch's day speech can decide when to surface / hide poison claims.
    try:
        from werewolf_agent.runtime.strategy.death import evaluate_death_cause_claims

        # Collect wolf kill target for this night from public events.
        wolf_kill_target_id: str | None = None
        for e in gs.events:
            if e.type == "wolf_kill_selected":
                night = e.payload.get("night_number", 0)
                target = e.payload.get("target_id", "")
                if night and target and night == gs.night_number:
                    wolf_kill_target_id = target
                    break

        evaluations = evaluate_death_cause_claims(
            gs, witch_id, "witch", wolf_kill_target_id=wolf_kill_target_id,
        )
        if evaluations:
            parts["witch_death_cause_evaluations"] = evaluations
    except Exception:
        # The directive must never break the agent's day speech; if the
        # strategy evaluator chokes (e.g., on partial state in tests),
        # silently skip the enrichment.
        pass

    return parts
