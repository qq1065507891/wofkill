# -*- coding: utf-8 -*-
"""
校验 7 月 14 日后审计问题的最小回归夹具目录。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-26
"""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from xml.etree import ElementTree


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
        "tests/runtime/test_wolf_prompt_contract.py::"
        "test_layered_context_rebuilds_live_status_instead_of_trusting_old_plan",
        "tests/runtime/test_wolf_prompt_contract.py::"
        "test_werewolf_system_prompt_states_target_and_evidence_semantics",
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
        "test_running_wolf_discussion_checkpoint_aborts_at_step_limit",
    ),
    "N4": (
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_reasoning_claim_cannot_override_structured_support_quorum",
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
        "tests/integration/test_post_july14_repair_closure.py::"
        "test_provider_failure_no_kill_event_has_complete_v2_audit_identity",
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


def _junit_closure_counts(report_path: Path) -> dict[str, int]:
    """读取子 pytest JUnit；skipped 不得计入真正执行数量。"""
    report = ElementTree.parse(report_path).getroot()
    summary = report if report.tag == "testsuite" else report.find("testsuite")
    if summary is None:
        raise AssertionError("mapped pytest JUnit has no testsuite")
    collected = int(summary.attrib.get("tests", "0"))
    skipped = int(summary.attrib.get("skipped", "0"))
    return {
        "collected": collected,
        "executed": max(0, collected - skipped),
        "failures": int(summary.attrib.get("failures", "0")),
        "errors": int(summary.attrib.get("errors", "0")),
        "skipped": skipped,
    }


def _junit_node_coverage(
    report_path: Path,
    expected_node_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """逐个确认显式映射函数至少产生一个 JUnit testcase。"""
    report = ElementTree.parse(report_path).getroot()
    testcases = tuple(report.findall(".//testcase"))
    observed: list[str] = []
    missing: list[str] = []
    for node_id in expected_node_ids:
        path_text, separator, function_name = node_id.partition("::")
        if not separator:
            missing.append(node_id)
            continue
        normalized_path = path_text.replace("\\", "/")
        if normalized_path.endswith(".py"):
            normalized_path = normalized_path[:-3]
        expected_classname = normalized_path.replace("/", ".")
        matched = any(
            testcase.attrib.get("classname") == expected_classname
            and (
                testcase.attrib.get("name") == function_name
                or str(testcase.attrib.get("name") or "").startswith(
                    f"{function_name}["
                )
            )
            for testcase in testcases
        )
        (observed if matched else missing).append(node_id)
    return {"observed": tuple(observed), "missing": tuple(missing)}


def _pytest_deselected_count(output: str) -> int:
    """从子 pytest 摘要读取 deselected，和缺失 JUnit node 分开报告。"""
    return sum(
        int(match.group(1))
        for match in re.finditer(r"\b(\d+)\s+deselected\b", output)
    )


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


def test_junit_closure_counts_do_not_treat_skipped_cases_as_executed(
    tmp_path,
) -> None:
    """参数实例数量不得掩盖未执行的显式映射节点。"""
    report_path = tmp_path / "skipped.xml"
    report_path.write_text(
        '<testsuite tests="3" failures="0" errors="0" skipped="1" />',
        encoding="utf-8",
    )

    counts = _junit_closure_counts(report_path)

    assert counts == {
        "collected": 3,
        "executed": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
    }


def test_junit_node_coverage_detects_missing_node_despite_parameter_instances(
    tmp_path,
) -> None:
    """一个参数节点的多个实例不能补偿另一个映射节点完全未执行。"""
    report_path = tmp_path / "parameter-mask.xml"
    report_path.write_text(
        """<testsuite tests="3" failures="0" errors="0" skipped="0">
        <testcase classname="tests.fake.test_one" name="test_param[zero]" />
        <testcase classname="tests.fake.test_one" name="test_param[one]" />
        <testcase classname="tests.fake.test_one" name="test_param[two]" />
        </testsuite>""",
        encoding="utf-8",
    )
    expected = (
        "tests/fake/test_one.py::test_param",
        "tests/fake/test_two.py::test_missing",
    )

    coverage = _junit_node_coverage(report_path, expected)

    assert coverage["observed"] == ("tests/fake/test_one.py::test_param",)
    assert coverage["missing"] == ("tests/fake/test_two.py::test_missing",)


def test_mapped_audit_nodes_execute_as_nonrecursive_closure_batch() -> None:
    """去重执行全部显式映射节点，禁止映射门禁递归调用自身。"""
    sentinel = "WOFKILL_TASK15_NODE_BATCH"
    assert os.environ.get(sentinel) != "1", "映射门禁发生递归执行"
    gate_node = (
        "tests/regression/test_post_july14_contract_cases.py::"
        "test_mapped_audit_nodes_execute_as_nonrecursive_closure_batch"
    )
    node_ids = tuple(dict.fromkeys(
        node_id
        for case_nodes in CASE_TEST_NODE_IDS.values()
        for node_id in case_nodes
    ))
    assert gate_node not in node_ids

    repository_root = Path(__file__).parents[2]
    temp_root = repository_root / ".tmp"
    temp_root.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment[sentinel] = "1"
    environment["PYTEST_ADDOPTS"] = ""
    with tempfile.TemporaryDirectory(
        prefix="task15-node-batch-", dir=temp_root
    ) as temporary:
        report_path = Path(temporary) / "mapped-nodes.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            *node_ids,
            "-q",
            "-n0",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(Path(temporary) / "pytest"),
            "--junitxml",
            str(report_path),
        ]
        completed = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )

        output = "\n".join((completed.stdout, completed.stderr)).strip()
        assert report_path.is_file(), output
        counts = _junit_closure_counts(report_path)
        coverage = _junit_node_coverage(report_path, node_ids)
        deselected = _pytest_deselected_count(output)

    assert counts["skipped"] == 0, output
    assert deselected == 0, f"mapped tests deselected={deselected}\n{output}"
    assert not coverage["missing"], (
        f"mapped pytest nodes missing from JUnit={coverage['missing']}\n{output}"
    )
    assert completed.returncode == 0, output
    assert counts["executed"] == counts["collected"], output
    assert counts["executed"] >= len(node_ids), (
        f"mapped tests executed={counts['executed']}, "
        f"explicit nodes={len(node_ids)}\n{output}"
    )
    assert counts["failures"] == 0 and counts["errors"] == 0, output
