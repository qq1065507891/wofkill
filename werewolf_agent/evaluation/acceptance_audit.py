# -*- coding: utf-8 -*-
"""
单次归一化后组合各验收领域投影器，并保留历史验收审计导入接口。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from typing import Any, Iterable

from werewolf_agent.evaluation.acceptance_power_metrics import (
    _friendly_fire_risk_complete,
    _power_role_evidence_complete,
    _retain_option_complete,
    _score,
    _signals,
    _target_evidence_complete,
    compute_power_acceptance_metrics,
)
from werewolf_agent.evaluation.acceptance_reflection_metrics import (
    _reflection_payload_matches_players,
    compute_reflection_acceptance_metrics,
)
from werewolf_agent.evaluation.acceptance_shared import (
    _game_player_roles,
    _is_non_negative_int,
    _non_negative_int,
)
from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
    _SEMANTIC_FALLBACK_KINDS,
    _semantic_identity,
    compute_terminal_semantic_acceptance_metrics,
)
from werewolf_agent.evaluation.acceptance_world_metrics import (
    compute_world_acceptance_metrics,
)
from werewolf_agent.evaluation.decision_execution_audit import (
    compute_decision_execution_metrics,
)
from werewolf_agent.evaluation.game_projection import (
    ensure_normalized_acceptance_games,
    projection_support,
)


def compute_acceptance_audit_metrics(
    games: Iterable[Any],
) -> dict[str, Any]:
    """组合执行 taxonomy 与跨责任域验收指标，供单局和批量报告复用。"""
    normalized = ensure_normalized_acceptance_games(games)
    supported, unsupported_reason = projection_support(normalized)
    return {
        **compute_decision_execution_metrics(normalized),
        **_compute_acceptance_metrics(normalized),
        "acceptance_projection_supported": supported,
        "acceptance_projection_unsupported_reason": unsupported_reason,
    }


def _compute_acceptance_metrics(games: Iterable[Any]) -> dict[str, Any]:
    """组合不依赖模型调用的四个验收领域投影结果。"""
    return {
        **compute_terminal_semantic_acceptance_metrics(games),
        **compute_world_acceptance_metrics(games),
        **compute_power_acceptance_metrics(games),
        **compute_reflection_acceptance_metrics(games),
    }


__all__ = [
    "compute_acceptance_audit_metrics",
    "_compute_acceptance_metrics",
    "_non_negative_int",
    "_is_non_negative_int",
    "_semantic_identity",
    "_game_player_roles",
    "_reflection_payload_matches_players",
    "_power_role_evidence_complete",
    "_target_evidence_complete",
    "_score",
    "_signals",
    "_friendly_fire_risk_complete",
    "_retain_option_complete",
    "_SEMANTIC_FALLBACK_KINDS",
]
