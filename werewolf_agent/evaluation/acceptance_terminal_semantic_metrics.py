# -*- coding: utf-8 -*-
"""
投影终局后调用与 V1/V2 语义修复验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-19
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from werewolf_agent.evaluation.acceptance_shared import (
    _is_non_negative_int,
    _non_negative_int,
)
from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    projection_support,
)
from werewolf_agent.agents.player_failures import (
    is_complete_terminal_fallback_v2,
)

_SEMANTIC_FALLBACK_KINDS = frozenset({
    "no_fallback",
    "generic_template",
    "target_specific",
    "verified_claim",
    "task_specific",
})


def compute_terminal_semantic_acceptance_metrics(
    games: Iterable[Any],
) -> dict[str, Any]:
    """扫描终局与语义事件并投影对应验收指标。"""
    return _compute_terminal_semantic_acceptance_metrics_from_normalized(
        normalize_acceptance_games(games)
    )


def _compute_terminal_semantic_acceptance_metrics_from_normalized(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """消费同一调用链刚完成验证的不可变游戏快照。"""
    projection_is_supported, projection_reason = projection_support(games)
    semantic_rows: list[dict[str, Any]] = []
    semantic_eligible_count = 0
    semantic_reconciliation_complete = True
    post_win_calls = 0
    terminal_fallback_count = 0
    terminal_failure_code_covered_count = 0
    terminal_fallback_invalid_count = 0
    terminal_fallback_kind_counts: Counter[str] = Counter()

    def record_terminal_fallback(
        row: Mapping[str, Any],
        *,
        source_type: str,
    ) -> None:
        """统一统计玩家动作与狼队计划的终退 V2 字段。"""
        nonlocal terminal_fallback_count, terminal_failure_code_covered_count
        nonlocal terminal_fallback_invalid_count
        is_candidate = (
            source_type == "wolf_team_plan_fallback"
            or row.get("generated_by") == "terminal_fallback"
            or row.get("decision_outcome") == "terminal_fallback"
        )
        if not is_candidate:
            return
        terminal_fallback_count += 1
        if not is_complete_terminal_fallback_v2(row, source_type=source_type):
            terminal_fallback_invalid_count += 1
            return
        terminal_failure_code_covered_count += 1
        fallback_kind = row.get("fallback_kind")
        if isinstance(fallback_kind, str) and fallback_kind:
            terminal_fallback_kind_counts[fallback_kind] += 1

    for game in games:
        victory_seen = False
        semantic_events_by_identity: dict[
            tuple[str, str, int, str], list[dict[str, Any]]
        ] = {}
        semantic_traces_by_identity: dict[
            tuple[str, str, int, str], list[Mapping[str, Any]]
        ] = {}
        for event in game.get("events", []):
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "victory":
                victory_seen = True
                continue
            if victory_seen and event_type in {
                "action_trace_audit",
                "model_execution_audit",
                "wolf_team_plan",
            }:
                task = str(payload.get("task_type") or payload.get("phase") or "")
                if task not in {"reflection", "post_game_reflection"}:
                    post_win_calls += 1
            if event_type == "wolf_team_plan_fallback":
                record_terminal_fallback(
                    payload,
                    source_type="wolf_team_plan_fallback",
                )
            if event_type == "semantic_repair_audit" and payload.get("repairable") is True:
                identity = _semantic_identity(payload)
                if identity is None:
                    semantic_reconciliation_complete = False
                else:
                    semantic_events_by_identity.setdefault(identity, []).append(payload)
            if event_type != "action_trace_audit":
                continue
            trace = payload.get("action_trace")
            if not isinstance(trace, Mapping):
                continue
            record_terminal_fallback(trace, source_type="action_trace_audit")
            semantic = trace.get("semantic_repair_audit")
            if isinstance(semantic, Mapping) and semantic.get("repairable") is True:
                semantic_eligible_count += 1
                identity = _semantic_identity(payload)
                if identity is None:
                    semantic_reconciliation_complete = False
                else:
                    semantic_traces_by_identity.setdefault(identity, []).append(semantic)

        for identity, traces in semantic_traces_by_identity.items():
            matches = semantic_events_by_identity.get(identity, [])
            if (
                len(matches) == 1
                and len(traces) == 1
                and _semantic_audit_rows_agree(matches[0], traces[0])
            ):
                semantic_rows.append(matches[0])
            else:
                semantic_reconciliation_complete = False
        if any(
            identity not in semantic_traces_by_identity or len(rows) != 1
            for identity, rows in semantic_events_by_identity.items()
        ):
            semantic_reconciliation_complete = False

    semantic_count = semantic_eligible_count
    semantic_reconciliation_valid = _semantic_reconciliation_is_valid(
        semantic_count,
        semantic_rows,
        semantic_reconciliation_complete,
    )
    semantic_source_complete = (
        projection_is_supported
        and semantic_reconciliation_valid
        and all(
            isinstance(row.get("success"), bool)
            and isinstance(row.get("target_preserved"), bool)
            and isinstance(row.get("speaker_attribution_preserved"), bool)
            and isinstance(row.get("negation_preserved"), bool)
            and _is_non_negative_int(row.get("introduced_claim_count"))
            and _has_supported_semantic_gate_version(row)
            and (
                not _is_semantic_gate_v2(row)
                or _is_non_negative_int(row.get("unsupported_public_claim_count"))
            )
            and row.get("fallback_kind") in _SEMANTIC_FALLBACK_KINDS
            and (
                (row.get("success") is True and row.get("fallback_kind") == "no_fallback")
                or (
                    row.get("success") is False
                    and row.get("fallback_kind") != "no_fallback"
                )
            )
            for row in semantic_rows
        )
    )
    semantic_success = sum(
        _semantic_row_is_successful(row)
        for row in semantic_rows
    )
    target_preserved = sum(row.get("target_preserved") is True for row in semantic_rows)
    speaker_attribution_preserved = sum(
        row.get("speaker_attribution_preserved") is True for row in semantic_rows
    )
    negation_preserved = sum(
        row.get("negation_preserved") is True for row in semantic_rows
    )
    no_new_claim = (
        sum(
            _non_negative_int(row.get("introduced_claim_count")) == 0
            for row in semantic_rows
        )
        if semantic_source_complete
        else 0
    )
    retention_source_complete = projection_is_supported and semantic_count > 0 and all(
        _is_non_negative_int(row.get("verified_claim_count"))
        and _is_non_negative_int(row.get("retained_verified_claim_count"))
        and row["retained_verified_claim_count"] <= row["verified_claim_count"]
        for row in semantic_rows
    )
    retention_rows = (
        [
            row
            for row in semantic_rows
            if _non_negative_int(row.get("verified_claim_count")) > 0
        ]
        if retention_source_complete
        else []
    )
    retained_verified_claims = sum(
        row["retained_verified_claim_count"] == row["verified_claim_count"]
        for row in retention_rows
    )
    generic_template_count = (
        sum(row.get("fallback_kind") == "generic_template" for row in semantic_rows)
        if semantic_source_complete
        else None
    )
    public_evidence_safety_supported = (
        projection_is_supported
        and semantic_reconciliation_valid
        and all(
            _is_semantic_gate_v2(row)
            and _is_non_negative_int(row.get("unsupported_public_claim_count"))
            for row in semantic_rows
        )
    )
    public_evidence_safe_count = sum(
        row.get("unsupported_public_claim_count") == 0
        for row in semantic_rows
    )
    terminal_metrics_supported = (
        projection_is_supported
        and terminal_fallback_count > 0
        and terminal_fallback_invalid_count == 0
    )
    return {
        "terminal_post_win_game_model_call_count": post_win_calls,
        "terminal_fallback_count": terminal_fallback_count,
        "terminal_fallback_original_failure_code_metrics_supported": (
            terminal_metrics_supported
        ),
        "terminal_fallback_original_failure_code_unsupported_reason": (
            projection_reason if not projection_is_supported
            else "no_terminal_fallback_observations"
            if terminal_fallback_count == 0
            else "incomplete_terminal_fallback_v2"
            if terminal_fallback_invalid_count
            else None
        ),
        "terminal_fallback_original_failure_code_covered_count": (
            terminal_failure_code_covered_count
        ),
        "terminal_fallback_original_failure_code_coverage_rate": (
            terminal_failure_code_covered_count / terminal_fallback_count
            if terminal_metrics_supported else None
        ),
        "terminal_fallback_kind_counts": dict(sorted(terminal_fallback_kind_counts.items())),
        "semantic_repair_metrics_supported": semantic_source_complete,
        "semantic_repair_metrics_unsupported_reason": (
            projection_reason if not projection_is_supported
            else None if semantic_source_complete else "incomplete_semantic_repair_audit"
        ),
        "semantic_repair_eligible_count": semantic_count,
        "semantic_repair_success_count": semantic_success,
        "semantic_repair_success_rate": (
            semantic_success / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_public_evidence_safety_metrics_supported": (
            public_evidence_safety_supported
        ),
        "semantic_repair_public_evidence_safety_rate": (
            public_evidence_safe_count / semantic_count
            if public_evidence_safety_supported
            else None
        ),
        "semantic_repair_target_preservation_rate": (
            target_preserved / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_speaker_attribution_preservation_rate": (
            speaker_attribution_preserved / semantic_count
            if semantic_source_complete else None
        ),
        "semantic_repair_negation_preservation_rate": (
            negation_preserved / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_no_new_claim_rate": (
            no_new_claim / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_generic_template_count": generic_template_count,
        "semantic_repair_verified_claim_retention_metrics_supported": (
            retention_source_complete and bool(retention_rows)
        ),
        "semantic_repair_verified_claim_retention_eligible_count": len(retention_rows),
        "semantic_repair_verified_claim_retained_count": retained_verified_claims,
        "semantic_repair_verified_claim_retention_rate": (
            retained_verified_claims / len(retention_rows) if retention_rows else None
        ),
    }


def _semantic_reconciliation_is_valid(
    semantic_count: int,
    semantic_rows: Sequence[Mapping[str, Any]],
    reconciliation_complete: bool,
) -> bool:
    """确认每条语义 trace 都有且仅有一条一致的 standalone 审计。"""
    return (
        reconciliation_complete
        and semantic_count > 0
        and len(semantic_rows) == semantic_count
    )


def _semantic_audit_rows_agree(
    standalone: Mapping[str, Any],
    nested: Mapping[str, Any],
) -> bool:
    """比较决定语义门控结果的成对审计字段。"""
    return all(
        standalone.get(field) == nested.get(field)
        for field in (
            "semantic_gate_version",
            "success",
            "speaker_attribution_preserved",
            "negation_preserved",
            "fallback_kind",
            "unsupported_public_claim_count",
        )
    )


def _has_supported_semantic_gate_version(row: Mapping[str, Any]) -> bool:
    """无版本行保持 V1 兼容；显式版本只能是 V2。"""
    return "semantic_gate_version" not in row or _is_semantic_gate_v2(row)


def _is_semantic_gate_v2(row: Mapping[str, Any]) -> bool:
    """判断语义审计行是否声明了受支持的 V2 门控。"""
    version = row.get("semantic_gate_version")
    return isinstance(version, int) and not isinstance(version, bool) and version == 2


def _semantic_row_is_successful(row: Mapping[str, Any]) -> bool:
    """按行版本计算成功，避免用 V1 观察指标否决 V2。"""
    if _is_semantic_gate_v2(row):
        return (
            row.get("success") is True
            and row.get("speaker_attribution_preserved") is True
            and row.get("negation_preserved") is True
            and _is_non_negative_int(row.get("unsupported_public_claim_count"))
            and row.get("unsupported_public_claim_count") == 0
        )
    return (
        row.get("success") is True
        and row.get("target_preserved") is True
        and row.get("speaker_attribution_preserved") is True
        and row.get("negation_preserved") is True
        and row.get("introduced_claim_count") == 0
        and row.get("retained_verified_claim_count") == row.get("verified_claim_count")
    )


def _semantic_identity(
    payload: dict[str, Any],
) -> tuple[str, str, int, str] | None:
    trace_id = payload.get("trace_id")
    game_id = payload.get("game_id")
    action_index = payload.get("action_index")
    task_type = payload.get("task_type")
    if (
        not isinstance(trace_id, str) or not trace_id
        or not isinstance(game_id, str) or not game_id
        or not isinstance(action_index, int) or isinstance(action_index, bool)
        or not isinstance(task_type, str) or not task_type
    ):
        return None
    return trace_id, game_id, action_index, task_type
