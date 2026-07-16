# -*- coding: utf-8 -*-
"""
统一解析狼人结算前空刀，并提供可复算的确定性恢复策略。

作者: Project contributors
创建日期: 2026-07-16

使用示例:
    >>> policy = NoKillPolicy(max_consecutive_pre_resolution_no_kill=2)
    >>> result = policy.resolve(game_state, reason_code="strategic_abstain")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.event_metadata import new_game_event


NoKillReasonCode = Literal[
    "strategic_abstain",
    "true_tie",
    "insufficient_quorum",
    "invalid_primary",
    "invalid_backup",
    "plan_generation_failed",
    "provider_unavailable",
]

NO_KILL_REASON_CODES = frozenset({
    "strategic_abstain",
    "true_tie",
    "insufficient_quorum",
    "invalid_primary",
    "invalid_backup",
    "plan_generation_failed",
    "provider_unavailable",
})


@dataclass(frozen=True)
class NoKillDecision:
    """所有空刀出口共享的稳定决策结构。"""

    reason_code: NoKillReasonCode
    consecutive_pre_resolution_no_kill_count: int
    forced_recovery_applied: bool
    recovered_target_id: str | None

    def to_payload(self) -> dict[str, Any]:
        """转换为 JSON 安全的事件载荷。"""
        return asdict(self)


class NoKillPolicy:
    """根据不可变游戏历史解析一次空刀，不持有跨局可变状态。"""

    def __init__(
        self,
        *,
        max_consecutive_pre_resolution_no_kill: int = 2,
    ) -> None:
        if (
            isinstance(max_consecutive_pre_resolution_no_kill, bool)
            or not isinstance(max_consecutive_pre_resolution_no_kill, int)
            or max_consecutive_pre_resolution_no_kill < 1
        ):
            raise ValueError(
                "max_consecutive_pre_resolution_no_kill must be a positive integer"
            )
        self._max_consecutive = max_consecutive_pre_resolution_no_kill

    def resolve(
        self,
        game_state: GameState,
        *,
        reason_code: NoKillReasonCode,
        event_type: str = "wolf_no_kill_timeout",
        primary_positive_support: Mapping[str, int] | None = None,
        backup_positive_support: Mapping[str, int] | None = None,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """生成统一空刀事件，超过阈值时确定性恢复合法狼刀。"""
        if reason_code not in NO_KILL_REASON_CODES:
            raise ValueError(f"unsupported no-kill reason code: {reason_code}")
        already_resolved, existing_target = _current_night_wolf_choice(game_state)
        if already_resolved:
            return {
                "game_state": game_state,
                "wolf_kill_target_id": existing_target,
            }

        prior_reasons = _consecutive_no_kill_reasons(game_state)
        count = len(prior_reasons) + 1
        if count <= self._max_consecutive:
            decision = NoKillDecision(
                reason_code=reason_code,
                consecutive_pre_resolution_no_kill_count=count,
                forced_recovery_applied=False,
                recovered_target_id=None,
            )
            event = new_game_event(
                game_state,
                event_type,
                {
                    "night_number": game_state.night_number,
                    "reason": reason_code,
                    **dict(extra_payload or {}),
                    "no_kill_decision": decision.to_payload(),
                },
                visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
            )
            return {
                "game_state": replace(
                    game_state,
                    events=[*game_state.events, event],
                ),
                "wolf_kill_target_id": None,
            }

        primary_support = dict(primary_positive_support or {})
        backup_support = dict(backup_positive_support or {})
        candidate_scores = _candidate_scores(
            game_state,
            primary_support=primary_support,
            backup_support=backup_support,
        )
        original_reasons = [*prior_reasons, reason_code]
        if not candidate_scores:
            decision = NoKillDecision(
                reason_code=reason_code,
                consecutive_pre_resolution_no_kill_count=count,
                forced_recovery_applied=True,
                recovered_target_id=None,
            )
            event = new_game_event(
                game_state,
                "forced_recovery_no_legal_target",
                {
                    "night_number": game_state.night_number,
                    "original_reasons": original_reasons,
                    "consecutive_pre_resolution_no_kill_count": count,
                    "candidate_scores": {},
                    "final_target_id": None,
                    "reason": reason_code,
                    "no_kill_decision": decision.to_payload(),
                },
                visibility=EventVisibility.MODERATOR_ONLY,
            )
            return {
                "game_state": replace(
                    game_state,
                    events=[*game_state.events, event],
                ),
                "wolf_kill_target_id": None,
            }

        target_id = max(candidate_scores, key=candidate_scores.__getitem__)
        decision = NoKillDecision(
            reason_code=reason_code,
            consecutive_pre_resolution_no_kill_count=count,
            forced_recovery_applied=True,
            recovered_target_id=target_id,
        )
        recovery = new_game_event(
            game_state,
            "wolf_kill_forced_recovery",
            {
                "night_number": game_state.night_number,
                "original_reasons": original_reasons,
                "consecutive_pre_resolution_no_kill_count": count,
                "candidate_scores": {
                    candidate_id: list(score)
                    for candidate_id, score in candidate_scores.items()
                },
                "final_target_id": target_id,
            },
            visibility=EventVisibility.MODERATOR_ONLY,
        )
        recovered_state = replace(
            game_state,
            events=[*game_state.events, recovery],
        )
        selected = new_game_event(
            recovered_state,
            "wolf_kill_selected",
            {
                "night_number": game_state.night_number,
                "target_id": target_id,
                "reason": "forced_recovery",
                "no_kill_decision": decision.to_payload(),
            },
            visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        )
        return {
            "game_state": replace(
                game_state,
                events=[*game_state.events, recovery, selected],
            ),
            "wolf_kill_target_id": target_id,
        }


def no_kill_policy_for_state(state: Mapping[str, Any]) -> NoKillPolicy:
    """从运行时 RuleEngine 读取阈值，缺省保持正式规则值 2。"""
    engine = state.get("engine")
    ruleset = getattr(engine, "ruleset", None)
    threshold = getattr(
        ruleset,
        "max_consecutive_pre_resolution_no_kill",
        2,
    )
    return NoKillPolicy(
        max_consecutive_pre_resolution_no_kill=threshold,
    )


def _consecutive_no_kill_reasons(game_state: GameState) -> list[NoKillReasonCode]:
    """倒序读取上次合法选刀之后的连续结算前空刀原因。"""
    reasons: list[NoKillReasonCode] = []
    seen_nights: set[int] = set()
    for reverse_index, event in enumerate(reversed(game_state.events)):
        if event.type == "wolf_kill_selected":
            break
        decision = event.payload.get("no_kill_decision")
        raw_reason = (
            decision.get("reason_code")
            if isinstance(decision, Mapping)
            else event.payload.get("reason")
        )
        if (
            event.type in {
                "wolf_no_kill_timeout",
                "wolf_no_kill_declared",
                "forced_recovery_no_legal_target",
            }
            and raw_reason in NO_KILL_REASON_CODES
        ):
            raw_night = event.payload.get("night_number")
            night_key = (
                raw_night
                if isinstance(raw_night, int) and not isinstance(raw_night, bool)
                else -(reverse_index + 1)
            )
            if night_key in seen_nights:
                continue
            seen_nights.add(night_key)
            reasons.append(raw_reason)
    reasons.reverse()
    return reasons


def _current_night_wolf_choice(
    game_state: GameState,
) -> tuple[bool, str | None]:
    """返回本夜已有选刀终态，重复调用不得追加事件或推进计数。"""
    for event in reversed(game_state.events):
        if (
            event.type == "wolf_kill_selected"
            and event.payload.get("night_number") == game_state.night_number
        ):
            target_id = event.payload.get("target_id")
            return True, target_id if isinstance(target_id, str) else None
        if (
            event.type in {
                "wolf_no_kill_timeout",
                "wolf_no_kill_declared",
                "forced_recovery_no_legal_target",
            }
            and event.payload.get("night_number") == game_state.night_number
        ):
            return True, None
    return False, None


def _candidate_scores(
    game_state: GameState,
    *,
    primary_support: Mapping[str, int],
    backup_support: Mapping[str, int],
) -> dict[str, tuple[int, int, int]]:
    """按主刀支持、备刀支持、负座位序号计算稳定评分。"""
    scores: dict[str, tuple[int, int, int]] = {}
    for seat_index, (player_id, player) in enumerate(
        game_state.players.items(),
        start=1,
    ):
        if not player.alive or player.role == "werewolf":
            continue
        scores[player_id] = (
            int(primary_support.get(player_id, 0)),
            int(backup_support.get(player_id, 0)),
            -seat_index,
        )
    return scores


__all__ = [
    "NO_KILL_REASON_CODES",
    "NoKillDecision",
    "NoKillPolicy",
    "NoKillReasonCode",
    "no_kill_policy_for_state",
]
