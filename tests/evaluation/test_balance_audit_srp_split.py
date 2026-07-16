# -*- coding: utf-8 -*-
"""
锁定平衡审计 SRP 拆分前后的兼容导入与关键输出。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations


def test_balance_audit_facade_reexports_split_metric_entrypoints() -> None:
    """旧 facade 导入应直接指向拆分后的唯一实现。"""
    from werewolf_agent.evaluation import balance_audit
    from werewolf_agent.evaluation.acceptance_audit import (
        _compute_acceptance_metrics,
        _non_negative_int,
        _power_role_evidence_complete,
        compute_acceptance_audit_metrics,
    )
    from werewolf_agent.evaluation.decision_execution_audit import (
        _critical_reasoning_status_metrics,
        _explicit_trace_task,
        _iter_action_traces,
        _iter_action_trace_records,
        _trace_actor,
        _trace_task,
        compute_decision_execution_metrics,
    )
    from werewolf_agent.evaluation.world_evidence_audit import (
        support_matches_world,
    )

    assert (
        balance_audit.compute_decision_execution_metrics
        is compute_decision_execution_metrics
    )
    assert (
        balance_audit.compute_acceptance_audit_metrics
        is compute_acceptance_audit_metrics
    )
    assert balance_audit._support_matches_world is support_matches_world
    assert balance_audit._compute_acceptance_metrics is _compute_acceptance_metrics
    assert balance_audit._non_negative_int is _non_negative_int
    assert balance_audit._power_role_evidence_complete is _power_role_evidence_complete
    assert (
        balance_audit._critical_reasoning_status_metrics
        is _critical_reasoning_status_metrics
    )
    assert balance_audit._iter_action_traces is _iter_action_traces
    assert balance_audit._iter_action_trace_records is _iter_action_trace_records
    assert balance_audit._trace_actor is _trace_actor
    assert balance_audit._trace_task is _trace_task
    assert balance_audit._explicit_trace_task is _explicit_trace_task


def test_split_metric_entrypoints_preserve_empty_input_contract() -> None:
    """空输入的 supported 与分母语义在迁移中保持不变。"""
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([])

    assert metrics["decision_execution_metrics_supported"] is False
    assert metrics["reasoning_confirmation_rate"] is None
    assert metrics["terminal_post_win_game_model_call_count"] == 0
    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["possible_world_metrics_supported"] is False
    assert metrics["power_role_evidence_metrics_supported"] is False
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_acceptance_facade_reexports_domain_helpers_without_duplicate_implementations() -> None:
    from werewolf_agent.evaluation import acceptance_audit
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        _power_role_evidence_complete,
    )
    from werewolf_agent.evaluation.acceptance_shared import _non_negative_int

    assert acceptance_audit._power_role_evidence_complete is (
        _power_role_evidence_complete
    )
    assert acceptance_audit._non_negative_int is _non_negative_int


def test_acceptance_facade_composes_all_domain_projectors(monkeypatch) -> None:
    from werewolf_agent.evaluation import acceptance_audit

    monkeypatch.setattr(
        acceptance_audit,
        "_compute_terminal_semantic_acceptance_metrics_from_normalized",
        lambda games: {"terminal_semantic_marker": len(games)},
    )
    monkeypatch.setattr(
        acceptance_audit,
        "_compute_world_acceptance_metrics_from_normalized",
        lambda games: {"world_marker": len(games)},
    )
    monkeypatch.setattr(
        acceptance_audit,
        "_compute_power_acceptance_metrics_from_normalized",
        lambda games: {"power_marker": len(games)},
    )
    monkeypatch.setattr(
        acceptance_audit,
        "_compute_reflection_acceptance_metrics_from_normalized",
        lambda games: {"reflection_marker": len(games)},
    )

    assert acceptance_audit._compute_acceptance_metrics([{}]) == {
        "terminal_semantic_marker": 1,
        "world_marker": 1,
        "power_marker": 1,
        "reflection_marker": 1,
    }
