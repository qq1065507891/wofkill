# -*- coding: utf-8 -*-
"""
功能描述：从人设派生的策略先验——规划阈值与语调的动态行为参数。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_DECEPTION_TASKS = {
    "speech",
    "sheriff_speech",
    "defense_speech",
    "wolf_discussion",
    "vote",
}


@dataclass(frozen=True)
class PersonaPolicyPrior:
    """Sanitized persona influence that cannot alter truth estimates."""

    risk_tolerance: float = 0.5
    aggression: float = 0.5
    deception_skill: float = 0.0
    leadership: float = 0.5
    vote_confidence_threshold_delta: float = 0.0
    claim_risk_threshold_delta: float = 0.0
    speech_directness: str = "medium"
    deception_allowed: bool = False

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any] | None,
        *,
        own_role: str,
        task_type: str,
    ) -> "PersonaPolicyPrior":
        params = _params(snapshot or {})
        risk = _clamp(_float(params.get("risk_tolerance"), 0.5))
        aggression = _clamp(_float(params.get("aggression"), 0.5))
        deception = _clamp(_float(params.get("deception_skill"), 0.0))
        leadership = _clamp(_float(params.get("leadership"), 0.5))

        vote_delta = ((0.5 - risk) * 0.12) + ((0.5 - aggression) * 0.08)
        vote_delta = _clamp_range(vote_delta, -0.15, 0.12)

        is_wolf = own_role == "werewolf"
        deception_allowed = is_wolf and task_type in _DECEPTION_TASKS and deception >= 0.4
        claim_delta = 0.0
        if deception_allowed:
            claim_delta = _clamp_range(((deception - 0.5) * 0.2) + ((risk - 0.5) * 0.1), 0.0, 0.16)

        directness_score = (aggression * 0.6) + (leadership * 0.25) + (risk * 0.15)
        if directness_score >= 0.68:
            directness = "high"
        elif directness_score <= 0.35:
            directness = "low"
        else:
            directness = "medium"

        return cls(
            risk_tolerance=risk,
            aggression=aggression,
            deception_skill=deception,
            leadership=leadership,
            vote_confidence_threshold_delta=round(vote_delta, 3),
            claim_risk_threshold_delta=round(claim_delta, 3),
            speech_directness=directness,
            deception_allowed=deception_allowed,
        )

    def vote_threshold(self, base: float = 0.7) -> float:
        return round(_clamp_range(base + self.vote_confidence_threshold_delta, 0.05, 0.95), 3)

    def claim_risk_threshold(self, base: float = 0.7) -> float:
        return round(_clamp_range(base + self.claim_risk_threshold_delta, 0.05, 0.95), 3)


def _params(snapshot: dict[str, Any]) -> dict[str, Any]:
    for key in ("effective_params", "base_params"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return _clamp_range(value, 0.0, 1.0)


def _clamp_range(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
