# -*- coding: utf-8 -*-
"""
验证近期平衡报告与审计闭环 PowerShell 脚本契约。

作者: Project contributors
修改日期: 2026-07-26
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_recent_balance_report_includes_new_guardrails(tmp_path):
    from scripts.analyze_recent_balance import build_recent_balance_report

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "seer"},
        },
        "events": [
            {"type": "sheriff_elected", "payload": {"sheriff_id": "p01"}},
            {
                "type": "action_trace",
                "payload": {
                    "phase": "sheriff_vote",
                    "action_trace": {"fallback_reason": "parse_error"},
                },
            },
            {
                "type": "wolf_team_plan_fallback",
                "payload": {"night_number": 1},
            },
            {
                "type": "wolf_team_plan",
                "payload": {"night_number": 1, "evidence_quality": "weak"},
            },
            {
                "type": "wolf_kill_selected",
                "payload": {
                    "night_number": 1,
                    "target_id": "p02",
                    "reason": "wolf_team_plan",
                },
            },
        ],
        "deaths": [{"player_id": "p02", "reason": "hunter_shot"}],
    }
    path = tmp_path / "game_g_recent.json"
    path.write_text(json.dumps(game), encoding="utf-8")

    report = build_recent_balance_report([path])

    assert report["persona_prompt_confirmation"] == {
        "supported": False,
        "configured_action_count": 0,
        "confirmed_action_count": 0,
        "confirmation_rate": None,
    }

    assert report["sheriff_werewolf_rate"] == 1.0
    assert report["sheriff_vote_fallback_rate"] == 1.0
    assert report["wolf_team_plan_fallback_rate"] == 1.0
    assert report["weak_plan_kill_rate"] == 1.0
    assert "hunter_friendly_fire_rate" in report


def test_soak_script_is_isolated_and_requires_explicit_exact_seeds() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_audit_closure_soak.ps1"

    text = script.read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)][int[]]$Seeds" in text
    assert "(714001..714010)" in text
    assert "$Seeds.Count -ne 10" in text
    assert "Select-Object -Unique" in text
    assert "$env:WEREWOLF_GAME_LOG_PATH" in text
    assert "$PSScriptRoot" in text
    assert "$runnerScript" in text
    assert "$analyzerScript" in text
    assert "$thresholdEvaluatorScript" in text
    assert "Push-Location" in text
    assert "try {" in text
    assert "finally {" in text
    assert "Pop-Location" in text
    assert "--output-dir $gameOutputDir" in text
    assert "$gameId = \"audit-$runId-seed-$seed\"" in text
    assert "--game-id $gameId" in text
    assert '$gameData.game_id -ne $gameId' in text
    assert "$gameIds.Add($gameId)" in text
    assert "audit-closure-report.json" in text
    assert "audit-closure-thresholds.json" in text
    assert "Test-Path -LiteralPath $thresholdPath" in text
    assert "audit-closure-soak-manifest.json" in text
    assert "finished_count" in text
    assert "aborted_count" in text


def test_soak_script_has_valid_powershell_ast() -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("需要 pwsh 或 powershell 才能验证 PowerShell AST")

    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_audit_closure_soak.ps1"
    command = (
        "$errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}', [ref]$null, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
