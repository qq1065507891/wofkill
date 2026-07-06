# -*- coding: utf-8 -*-
"""
合并和裁剪 Agent strategy_directive。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.context_strategy_directives import _cap_strategy_directive
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.directive_priority import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
)

_MAX_STRATEGY_DIRECTIVE_TOKENS = 1500
_ROUND_SPECIFIC_DROP_KEYS: tuple[str, ...] = (
    "sheriff_election_record",
    "day_discussion_summary",
    "vote_pressure_context",
    "vote_history",
    "skill_tactical_advice",
    "death_cause_evaluation",
    "witch_death_cause_evaluations",
    "wolf_teammate_exposed",
    "belief_state",
    "must_address",
)


def _merge_strategy_directive(
    context: Any,
    new_directive: dict[str, Any],
) -> Any:
    """把新策略指令合并到上下文，并执行大小裁剪。"""

    existing = context.strategy_directive or {}
    merged: dict[str, Any] = {**existing, **new_directive}
    merged = _cap_strategy_directive(merged)
    return context.model_copy(update={"strategy_directive": merged})


def _directive_size(directive: dict[str, Any]) -> int:
    """粗略估算 strategy_directive 的 token 数。"""

    total = 0
    for value in directive.values():
        try:
            total += len(str(value))
        except Exception:
            continue
    return total // 2


def _cap_strategy_directive(
    directive: dict[str, Any],
    cap_tokens: int = _MAX_STRATEGY_DIRECTIVE_TOKENS,
) -> dict[str, Any]:
    """优先丢弃低价值回合级字段，直到 strategy_directive 不超过上限。"""

    if _directive_size(directive) <= cap_tokens:
        return directive
    ordered_candidates: list[str] = []
    for key in _ROUND_SPECIFIC_DROP_KEYS:
        if key in directive and key not in HARD_CONSTRAINT_KEYS:
            ordered_candidates.append(key)
    for key in directive:
        if key in REFERENCE_KEYS and key not in ordered_candidates:
            ordered_candidates.append(key)
    for key in directive:
        if (
            key not in HARD_CONSTRAINT_KEYS
            and key not in SUGGESTION_KEYS
            and key not in REFERENCE_KEYS
            and key not in ordered_candidates
        ):
            ordered_candidates.append(key)

    for key in ordered_candidates:
        if _directive_size(directive) <= cap_tokens:
            break
        directive = {k: v for k, v in directive.items() if k != key}
    return directive
