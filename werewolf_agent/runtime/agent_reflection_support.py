# -*- coding: utf-8 -*-
"""
提供赛后 Agent 反思上下文裁剪和执行支持。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.agent_reflection_support import _strip_in_game_directives
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import FallbackAction, TaskType
from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.context import _merge_strategy_directive, build_agent_context
from werewolf_agent.runtime.reflection_prompt import build_reflection_prompt

logger = logging.getLogger(__name__)

_POST_GAME_KEEP = frozenset({"reflection_task", "game_outcome"})


def _strip_in_game_directives(context):
    """赛后反思：剥离赛内决策 directive，只保留 allowlist。"""

    kept = {
        key: value
        for key, value in (context.strategy_directive or {}).items()
        if key in _POST_GAME_KEEP
    }
    return context.model_copy(update={"strategy_directive": kept})


def _terminal_reflection_verification(
    action: Any,
) -> dict[str, Any] | None:
    """把反思终退显式标成未生成，禁止把兜底文本当作经验。"""
    if not isinstance(action, FallbackAction) or action.trace is None:
        return None
    trace = action.trace
    if (
        trace.generated_by != "terminal_fallback"
        or trace.fallback_kind != "reflection_not_generated"
    ):
        return None
    return {
        "status": "not_generated",
        "failure_code": trace.original_failure_code or trace.terminal_failure_code,
        "failure_stage": trace.failure_stage,
        "verified_fact_count": 0,
        "verified_claim_ids": [],
        "rejected_claim_ids": [],
        "verified_lessons": [],
        "rejected_fact_count": 0,
        "rejected_lesson_count": 0,
    }


def _agent_reflection(
    state: dict[str, Any],
    engine: Any,
    registry: Any,
    player_id: str,
) -> dict[str, Any]:
    """赛后反思：让每名玩家复盘自己的表现。"""

    agent = registry.get_agent(player_id)
    if agent is None:
        return {}

    gs: GameState = state["game_state"]
    player = gs.players.get(player_id)
    winner = gs.winning_faction or "?"

    try:
        try:
            from werewolf_agent.runtime import agent_adapter as adapter_mod
            build_context = getattr(adapter_mod, "build_agent_context", build_agent_context)
            merge_directive = getattr(adapter_mod, "_merge_strategy_directive", _merge_strategy_directive)
        except Exception:
            build_context = build_agent_context
            merge_directive = _merge_strategy_directive

        context = build_context(
            engine, gs, player_id, TaskType.REFLECTION,
            legal_actions=[],
            restored_memory=state.get("restored_memory"),
            cognition_state_manager=state.get("cognition_state_manager"),
        )
        reflection_task = build_reflection_prompt(
            player=player,
            winner=winner,
            hybrid_master_faction=gs.hybrid_master_faction,
            state=gs,
        )
        reflection_directive = {
            "reflection_task": reflection_task,
            "game_outcome": (
                f"胜利方是{'好人' if winner == 'good' else '狼人'}阵营。"
                f"你{'存活到' if (player and player.alive) else '在'}游戏结束。"
                f"你的身份是 {player.role if player else '?'}。"
            ),
        }
        context = _strip_in_game_directives(context)
        context = merge_directive(context, reflection_directive)

        from werewolf_agent.memory.reflection_sanitization import (
            anonymize_player_ids_recursive,
        )
        from werewolf_agent.memory.reflection_synthesis import (
            verify_reflection_draft,
        )

        draft = agent.generate_reflection(context, reflection_task)
        verification = verify_reflection_draft(draft, gs)
        return {"reflection_verification": {
            "status": "verified",
            "decision_id": f"reflection:{gs.game_id}:{player_id}",
            "verified_fact_count": len(verification.verified_claims),
            "verified_claim_ids": [
                claim.claim_id for claim in verification.verified_claims
            ],
            "rejected_claim_ids": [
                claim.claim_id for claim in draft.claims
                if claim.claim_id not in {
                    verified.claim_id for verified in verification.verified_claims
                }
            ],
            "verified_lessons": anonymize_player_ids_recursive([
                {
                    "lesson_id": lesson.lesson_id,
                    "abstraction": lesson.abstraction,
                }
                for lesson in verification.verified_lessons
            ]),
            "rejected_fact_count": verification.rejected_fact_count,
            "rejected_lesson_count": verification.rejected_lesson_count,
        }}
    except Exception as exc:
        failure_code = str(getattr(exc, "failure_code", "agent_error"))
        if failure_code == "invalid_structured_draft":
            return {"reflection_verification": {
                "status": "invalid_structured_draft",
                "failure_code": failure_code,
                "failure_stage": "schema_validated",
                "verified_fact_count": 0,
                "verified_claim_ids": [],
                "rejected_claim_ids": [],
                "verified_lessons": [],
                "rejected_fact_count": 0,
                "rejected_lesson_count": 0,
            }}
        logger.warning(
            "Reflection generation failed for %s code=%s",
            player_id,
            failure_code,
        )
        return {"reflection_verification": {
            "status": "agent_error",
            "failure_code": failure_code,
            "failure_stage": "generated",
            "verified_fact_count": 0,
            "verified_claim_ids": [],
            "rejected_claim_ids": [],
            "verified_lessons": [],
            "rejected_fact_count": 0,
            "rejected_lesson_count": 0,
        }}
