# -*- coding: utf-8 -*-
"""
统一解析狼人结算前空刀，并提供可复算的确定性恢复策略。

作者: Project contributors
创建日期: 2026-07-16
修改日期: 2026-07-20

使用示例:
    >>> policy = NoKillPolicy(max_consecutive_pre_resolution_no_kill=2)
    >>> result = policy.resolve(game_state, reason_code="strategic_abstain")
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.event_metadata import (
    validate_v2_event_identity,
    validate_v2_event_log_identity,
)
from werewolf_agent.runtime.wolf_decision_trace import new_wolf_decision_event


logger = logging.getLogger("werewolf_agent.runtime.wolf_no_kill_policy")


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

_RESERVED_NO_KILL_PAYLOAD_KEYS = frozenset({
    "night_number",
    "reason",
    "reason_code",
    "no_kill_decision",
    "target_id",
    "original_reasons",
    "consecutive_pre_resolution_no_kill_count",
    "forced_recovery_applied",
    "recovered_target_id",
    "candidate_scores",
    "final_target_id",
})
_V2_IDENTITY_FIELDS = (
    "schema_version",
    "event_id",
    "sequence_number",
    "occurred_at",
    "game_id",
)


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
        reserved = _RESERVED_NO_KILL_PAYLOAD_KEYS.intersection(
            (extra_payload or {}).keys()
        )
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"reserved no-kill payload keys: {names}")
        already_resolved, existing_target = _current_night_wolf_choice(game_state)
        if already_resolved:
            logger.debug(
                "  [狼人决策] NoKillPolicy.resolve 入口: reason=%s already_resolved=%s existing_target=%s",
                reason_code,
                already_resolved,
                existing_target,
            )
            return {
                "game_state": game_state,
                "wolf_kill_target_id": existing_target,
            }

        prior_reasons = _consecutive_no_kill_reasons(game_state)
        count = len(prior_reasons) + 1
        logger.debug(
            "  [狼人决策] NoKillPolicy.resolve: reason=%s prior_reasons=%s count=%d threshold=%d",
            reason_code,
            list(prior_reasons),
            count,
            self._max_consecutive,
        )
        if count <= self._max_consecutive:
            decision = NoKillDecision(
                reason_code=reason_code,
                consecutive_pre_resolution_no_kill_count=count,
                forced_recovery_applied=False,
                recovered_target_id=None,
            )
            event = new_wolf_decision_event(
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
            event = new_wolf_decision_event(
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
        recovery = new_wolf_decision_event(
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
        selected = new_wolf_decision_event(
            recovered_state,
            "wolf_kill_selected",
            {
                "night_number": game_state.night_number,
                "target_id": target_id,
                "reason": "forced_recovery",
                "no_kill_decision": decision.to_payload(),
            },
            visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
            action_index=1,
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


@dataclass(frozen=True)
class _NormalizedLegacyWolfChoice:
    """V1 检查点中经白名单归一化的狼队夜间选择。"""

    kind: Literal["no_kill", "selected"]
    night_number: int
    reason_code: NoKillReasonCode | None = None
    target_id: str | None = None


def _has_v2_identity(event: GameEvent) -> bool:
    """判断事件是否声明了任一 V2 身份字段。"""
    return any(getattr(event, field) is not None for field in _V2_IDENTITY_FIELDS)


def _positive_night_number(event: GameEvent) -> int | None:
    raw_night = event.payload.get("night_number")
    if (
        not isinstance(raw_night, int)
        or isinstance(raw_night, bool)
        or raw_night < 1
    ):
        return None
    return raw_night


def _legacy_visibility_is_safe(
    event: GameEvent,
    *,
    allowed: set[EventVisibility],
) -> bool:
    """旧事件可无可见性字段；若携带则必须符合事件语义。"""
    raw_visibility = event.visibility or event.payload.get("visibility")
    if raw_visibility is None:
        return True
    aliases = {
        "moderator": EventVisibility.MODERATOR_ONLY,
        "wolf_team": EventVisibility.WEREWOLF_TEAM_ONLY,
    }
    try:
        visibility = (
            raw_visibility
            if isinstance(raw_visibility, EventVisibility)
            else (
                aliases[raw_visibility]
                if raw_visibility in aliases
                else EventVisibility(str(raw_visibility))
            )
        )
    except (TypeError, ValueError):
        return False
    return visibility in allowed


def _normalize_legacy_wolf_choice(
    game_state: GameState,
    event: GameEvent,
    *,
    require_alive_target: bool,
) -> _NormalizedLegacyWolfChoice | None:
    """仅适配已知 V1 形态，含任何 V2 身份的事件不得降级读取。"""
    if _has_v2_identity(event):
        return None
    night_number = _positive_night_number(event)
    if night_number is None:
        return None

    if event.type == "wolf_kill_selected":
        if not _legacy_visibility_is_safe(
            event,
            allowed={EventVisibility.WEREWOLF_TEAM_ONLY},
        ):
            return None
        target_id = event.payload.get("target_id")
        target = game_state.players.get(target_id) if isinstance(target_id, str) else None
        if (
            target is None
            or target.role == "werewolf"
            or (require_alive_target and not target.alive)
        ):
            return None
        return _NormalizedLegacyWolfChoice(
            kind="selected",
            night_number=night_number,
            target_id=target_id,
        )

    if event.type == "wolf_no_kill_timeout":
        if not _legacy_visibility_is_safe(
            event,
            allowed={EventVisibility.WEREWOLF_TEAM_ONLY},
        ):
            return None
        raw_reason = event.payload.get("reason")
        reason = (
            raw_reason
            if raw_reason in NO_KILL_REASON_CODES
            else "provider_unavailable"
        )
    elif event.type == "wolf_no_kill_declared":
        if not _legacy_visibility_is_safe(
            event,
            allowed={EventVisibility.WEREWOLF_TEAM_ONLY},
        ):
            return None
        raw_reason = event.payload.get("reason")
        reason = (
            raw_reason
            if raw_reason in NO_KILL_REASON_CODES
            else "strategic_abstain"
        )
    elif event.type == "wolf_plan_invalid_no_kill":
        if not _legacy_visibility_is_safe(
            event,
            allowed={
                EventVisibility.WEREWOLF_TEAM_ONLY,
                EventVisibility.MODERATOR_ONLY,
            },
        ):
            return None
        reason = "plan_generation_failed"
    elif event.type == "timer_expired":
        timer_key = event.payload.get("timer_key", event.payload.get("phase"))
        if timer_key not in {"wolf_discussion", "wolf_consensus", "wolf_team_plan"}:
            return None
        reason = "provider_unavailable"
    else:
        return None

    return _NormalizedLegacyWolfChoice(
        kind="no_kill",
        night_number=night_number,
        reason_code=reason,
    )


def _valid_no_kill_decision(
    value: Any,
    *,
    expected_reason: Any,
    expected_forced: bool,
) -> bool:
    """验证 V2 空刀终态携带完整且自洽的 NoKillDecision。"""
    if not isinstance(value, Mapping) or set(value) != {
        "reason_code",
        "consecutive_pre_resolution_no_kill_count",
        "forced_recovery_applied",
        "recovered_target_id",
    }:
        return False
    count = value.get("consecutive_pre_resolution_no_kill_count")
    return bool(
        expected_reason in NO_KILL_REASON_CODES
        and value.get("reason_code") == expected_reason
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 1
        and value.get("forced_recovery_applied") is expected_forced
        and value.get("recovered_target_id") is None
    )


def _trusted_v2_wolf_choice(
    game_state: GameState,
    event: GameEvent,
    *,
    require_alive_target: bool,
) -> _NormalizedLegacyWolfChoice | None:
    """验证单个 V2 狼队选择事件的身份、可见性和业务 schema。"""
    night_number = _positive_night_number(event)
    if night_number is None:
        return None
    required_visibility = (
        EventVisibility.MODERATOR_ONLY
        if event.type == "forced_recovery_no_legal_target"
        else EventVisibility.WEREWOLF_TEAM_ONLY
    )
    try:
        validate_v2_event_identity(
            game_state.game_id,
            event,
            required_visibility=required_visibility,
        )
    except ValueError:
        return None

    if event.type == "wolf_kill_selected":
        target_id = event.payload.get("target_id")
        target = game_state.players.get(target_id) if isinstance(target_id, str) else None
        if (
            target is None
            or target.role == "werewolf"
            or (require_alive_target and not target.alive)
        ):
            return None
        decision = event.payload.get("no_kill_decision")
        is_forced_recovery = event.payload.get("reason") == "forced_recovery"
        if "no_kill_decision" in event.payload or is_forced_recovery:
            if not (
                is_forced_recovery
                and isinstance(decision, Mapping)
                and decision.get("reason_code") in NO_KILL_REASON_CODES
                and decision.get("forced_recovery_applied") is True
                and decision.get("recovered_target_id") == target_id
                and isinstance(
                    decision.get("consecutive_pre_resolution_no_kill_count"),
                    int,
                )
                and not isinstance(
                    decision.get("consecutive_pre_resolution_no_kill_count"),
                    bool,
                )
                and decision.get("consecutive_pre_resolution_no_kill_count") >= 1
                and set(decision) == {
                    "reason_code",
                    "consecutive_pre_resolution_no_kill_count",
                    "forced_recovery_applied",
                    "recovered_target_id",
                }
            ):
                return None
        return _NormalizedLegacyWolfChoice(
            kind="selected",
            night_number=night_number,
            target_id=target_id,
        )

    if event.type not in {
        "wolf_no_kill_timeout",
        "wolf_no_kill_declared",
        "forced_recovery_no_legal_target",
    }:
        return None
    expected_forced = event.type == "forced_recovery_no_legal_target"
    reason = event.payload.get("reason")
    if not _valid_no_kill_decision(
        event.payload.get("no_kill_decision"),
        expected_reason=reason,
        expected_forced=expected_forced,
    ):
        return None
    return _NormalizedLegacyWolfChoice(
        kind="no_kill",
        night_number=night_number,
        reason_code=reason,
    )


def _normalized_wolf_choice(
    game_state: GameState,
    event: GameEvent,
    *,
    require_alive_target: bool,
) -> _NormalizedLegacyWolfChoice | None:
    """按 V2 严格读取或显式 V1 适配返回统一选择。"""
    if _has_v2_identity(event):
        return _trusted_v2_wolf_choice(
            game_state,
            event,
            require_alive_target=require_alive_target,
        )
    return _normalize_legacy_wolf_choice(
        game_state,
        event,
        require_alive_target=require_alive_target,
    )


def _consecutive_no_kill_reasons(game_state: GameState) -> list[NoKillReasonCode]:
    """倒序读取上次合法选刀之后的连续结算前空刀原因。"""
    reasons: list[NoKillReasonCode] = []
    seen_nights: set[int] = set()
    try:
        validate_v2_event_log_identity(game_state.game_id, game_state.events)
        trust_v2_log = True
    except ValueError:
        trust_v2_log = False
    for event in reversed(game_state.events):
        if _has_v2_identity(event) and not trust_v2_log:
            continue
        choice = _normalized_wolf_choice(
            game_state,
            event,
            require_alive_target=False,
        )
        if choice is None:
            continue
        if choice.kind == "selected":
            break
        if choice.night_number in seen_nights or choice.reason_code is None:
            continue
        seen_nights.add(choice.night_number)
        reasons.append(choice.reason_code)
    reasons.reverse()
    return reasons


def _current_night_wolf_choice(
    game_state: GameState,
) -> tuple[bool, str | None]:
    """返回本夜已有选刀终态，重复调用不得追加事件或推进计数。"""
    try:
        validate_v2_event_log_identity(game_state.game_id, game_state.events)
        trust_v2_log = True
    except ValueError:
        trust_v2_log = False
    for event in reversed(game_state.events):
        if _has_v2_identity(event) and not trust_v2_log:
            continue
        choice = _normalized_wolf_choice(
            game_state,
            event,
            require_alive_target=False,
        )
        if choice is None or choice.night_number != game_state.night_number:
            continue
        if choice.kind == "selected":
            return True, choice.target_id
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
