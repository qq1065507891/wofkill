# -*- coding: utf-8 -*-
"""
校验 7 月 14 日后审计问题的最小回归夹具目录。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-18
"""

import ast
import json
from pathlib import Path


CASE_TEST_NODE_IDS: dict[str, tuple[str, ...]] = {
    "K1": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_majority_kill_saved_by_witch_is_skill_cancellation_not_no_kill",
    ),
    "K2": (
        "tests/runtime/test_wolf_prompt_contract.py::"
        "test_layered_context_preserves_all_stances_and_budgets_only_raw_text",
    ),
    "K3": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_invalid_primary_uses_independent_majority_backup",
    ),
    "N1": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_single_wolf_primary_executes_without_tie",
    ),
    "N2": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_third_pre_resolution_no_kill_recovers_deterministically",
    ),
    "N3": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_graph_recursion_abort_persists_minimal_json",
    ),
    "N4": (
        "tests/integration/test_wolf_team_plan_e2e.py::"
        "test_e2e_n1_authoritative_stances_override_llm_recommendation",
    ),
    "N5": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_majority_kill_saved_by_witch_is_skill_cancellation_not_no_kill",
    ),
    "N6": (
        "tests/evaluation/test_game_balance_batch.py::"
        "test_balance_audit_reports_disjoint_wolf_plan_outcomes_with_exact_denominator",
    ),
    "N7": (
        "tests/runtime/test_event_metadata_v2.py::"
        "test_stamp_new_events_assigns_stable_v2_metadata",
    ),
    "N8": (
        "tests/runtime/test_reflection_transaction.py::"
        "test_zero_expected_entries_emits_no_valid_entries_and_never_succeeds",
    ),
    "N9": (
        "tests/runtime/test_resolution_batches.py::"
        "test_parse_legacy_resolution_batch",
    ),
    "N10": (
        "tests/runtime/test_speech_quality.py::"
        "test_non_empty_terminal_fallback_does_not_count_as_model_success",
    ),
    "N11": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_final_quality_distinguishes_valid_and_invalid_reflection",
    ),
    "N12": (
        "tests/model_gateway/test_provider_fallback_policy.py::"
        "test_route_identity_uses_provider_and_model_pair",
    ),
}


def load_cases() -> dict[str, dict[str, object]]:
    """读取审计问题的脱敏回归用例目录。"""
    fixture_path = Path(__file__).parents[1] / "fixtures" / "post_july14_contract_regressions.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_every_audit_issue_has_a_regression_case() -> None:
    cases = load_cases()
    assert set(cases) == {"K1", "K2", "K3", *(f"N{i}" for i in range(1, 13))}
    assert all(case["expected_contract"] for case in cases.values())


def test_every_audit_issue_maps_to_an_existing_pytest_node() -> None:
    """保证 15 项问题都指向可收集的测试函数，而不是自由文本。"""
    expected = {"K1", "K2", "K3", *(f"N{i}" for i in range(1, 13))}
    missing_cases = sorted(expected - CASE_TEST_NODE_IDS.keys())
    extra_cases = sorted(CASE_TEST_NODE_IDS.keys() - expected)
    empty_cases = sorted(
        case_id
        for case_id, node_ids in CASE_TEST_NODE_IDS.items()
        if not node_ids
    )
    assert not (missing_cases or extra_cases or empty_cases), (
        "审计问题测试映射不完整: "
        f"missing={missing_cases}, extra={extra_cases}, empty={empty_cases}"
    )

    repository_root = Path(__file__).parents[2]
    invalid_nodes: list[str] = []
    for node_ids in CASE_TEST_NODE_IDS.values():
        for node_id in node_ids:
            path_text, separator, function_name = node_id.partition("::")
            path = repository_root / path_text
            if not separator or not function_name or not path.is_file():
                invalid_nodes.append(node_id)
                continue
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            function_names = {
                node.name
                for node in ast.walk(module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if function_name not in function_names:
                invalid_nodes.append(node_id)

    assert not invalid_nodes, f"审计问题映射到不存在的 pytest node: {invalid_nodes}"
