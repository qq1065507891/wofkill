# -*- coding: utf-8 -*-
"""
单次归一化后组合各验收领域投影器，并保留历史验收审计导入接口。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from werewolf_agent.evaluation.acceptance_power_metrics import (
    _compute_power_acceptance_metrics_from_normalized,
    _friendly_fire_risk_complete,
    _power_role_evidence_complete,
    _retain_option_complete,
    _score,
    _signals,
    _target_evidence_complete,
)
from werewolf_agent.evaluation.acceptance_reflection_metrics import (
    _compute_reflection_acceptance_metrics_from_normalized,
    _reflection_payload_matches_players,
)
from werewolf_agent.evaluation.acceptance_shared import (
    _game_player_roles,
    _is_non_negative_int,
    _non_negative_int,
)
from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
    _compute_terminal_semantic_acceptance_metrics_from_normalized,
    _SEMANTIC_FALLBACK_KINDS,
    _semantic_identity,
)
from werewolf_agent.evaluation.acceptance_world_metrics import (
    _compute_world_acceptance_metrics_from_normalized,
)
from werewolf_agent.evaluation.decision_execution_audit import (
    compute_decision_execution_metrics,
)
from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    projection_support,
)


def compute_acceptance_audit_metrics(
    games: Iterable[Any],
) -> dict[str, Any]:
    """组合执行 taxonomy 与跨责任域验收指标，供单局和批量报告复用。"""
    return _compute_acceptance_audit_metrics_from_normalized(
        normalize_acceptance_games(games)
    )


def _compute_acceptance_audit_metrics_from_normalized(
    normalized: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """组合本次调用刚验证的快照，不作为外部输入安全边界。"""
    supported, unsupported_reason = projection_support(normalized)
    return {
        **compute_decision_execution_metrics(normalized),
        **_compute_acceptance_metrics_from_normalized(normalized),
        "acceptance_projection_supported": supported,
        "acceptance_projection_unsupported_reason": unsupported_reason,
    }


def _compute_acceptance_metrics(games: Iterable[Any]) -> dict[str, Any]:
    """验证公开输入后组合不依赖模型调用的四个验收领域指标。"""
    return _compute_acceptance_metrics_from_normalized(
        normalize_acceptance_games(games)
    )


def _compute_acceptance_metrics_from_normalized(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """组合本次调用刚验证的四领域快照，不接受外部可信标志。"""
    return {
        **_compute_terminal_semantic_acceptance_metrics_from_normalized(games),
        **_compute_world_acceptance_metrics_from_normalized(games),
        **_compute_power_acceptance_metrics_from_normalized(games),
        **_compute_reflection_acceptance_metrics_from_normalized(games),
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
