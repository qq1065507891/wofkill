# -*- coding: utf-8 -*-
"""
校验 7 月 14 日后审计问题的最小回归夹具目录。

作者: Project contributors
创建日期: 2026-07-15
"""

import json
from pathlib import Path


def load_cases() -> dict[str, dict[str, object]]:
    """读取审计问题的脱敏回归用例目录。"""
    fixture_path = Path(__file__).parents[1] / "fixtures" / "post_july14_contract_regressions.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_every_audit_issue_has_a_regression_case() -> None:
    cases = load_cases()
    assert set(cases) == {"K1", "K2", "K3", *(f"N{i}" for i in range(1, 13))}
    assert all(case["expected_contract"] for case in cases.values())
