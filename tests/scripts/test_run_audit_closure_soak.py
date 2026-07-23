# -*- coding: utf-8 -*-
"""
离线验证审查闭环 soak PowerShell 脚本的真实命令编排与状态恢复。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOAK_SCRIPT = REPO_ROOT / "scripts" / "run_audit_closure_soak.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell.exe")


@pytest.fixture
def fake_python(tmp_path: Path) -> tuple[Path, Path]:
    """创建只记录调用且绝不加载真实游戏代码的 Python 命令。"""

    dispatcher = tmp_path / "fake_python_dispatcher.py"
    dispatcher.write_text(
        """# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import subprocess
import sys

script_name = Path(sys.argv[1]).name
args = sys.argv[2:]
log_path = Path(os.environ["FAKE_PYTHON_CALL_LOG"])
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"script": script_name, "args": args}) + "\\n")

if script_name == "run_real_game.py":
    call_number = sum(
        1
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["script"] == "run_real_game.py"
    )
    if call_number == int(os.environ.get("FAKE_RUNNER_FAIL_AT", "0")):
        print("synthetic runner failure", file=sys.stderr)
        raise SystemExit(17)
    values = {args[index]: args[index + 1] for index in range(0, len(args), 2)}
    output_dir = Path(values["--output-dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path_mode = os.environ.get("FAKE_RUNNER_PATH_MODE", "inside")
    if path_mode == "outside":
        outside_dir = output_dir.parent / f"escaped-{call_number}"
        outside_dir.mkdir(parents=True, exist_ok=True)
        game_path = outside_dir / "game.json"
    elif path_mode == "symlink_escape":
        outside_dir = output_dir.parent / f"escaped-{call_number}"
        outside_dir.mkdir(parents=True, exist_ok=True)
        link_dir = output_dir / "linked"
        try:
            link_dir.symlink_to(outside_dir, target_is_directory=True)
        except OSError:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_dir), str(outside_dir)],
                check=True,
                capture_output=True,
            )
        game_path = link_dir / "game.json"
    else:
        game_path = output_dir / "game.json"
    game_path.write_text(
        json.dumps({
            "game_id": values["--game-id"],
            "seed": int(values["--seed"]),
            "status": "aborted" if call_number == int(
                os.environ.get("FAKE_ABORT_AT", "0")
            ) else "finished",
            "winning_faction": None if call_number == int(
                os.environ.get("FAKE_ABORT_AT", "0")
            ) else "good",
            "termination_reason": "step_limit" if call_number == int(
                os.environ.get("FAKE_ABORT_AT", "0")
            ) else None,
        }),
        encoding="utf-8",
    )
    print(f"Game log: {game_path.resolve()}")
    if call_number == int(os.environ.get("FAKE_ABORT_AT", "0")):
        raise SystemExit(1)
elif script_name == "analyze_recent_balance.py":
    print(json.dumps({"completion_rate": 1.0, "input_count": len(args)}))
elif script_name == "evaluate_audit_closure_thresholds.py":
    threshold_path = Path(args[1])
    threshold_path.write_text(
        json.dumps({"overall_pass": not bool(os.environ.get("FAKE_THRESHOLD_FAIL"))}),
        encoding="utf-8",
    )
    if os.environ.get("FAKE_THRESHOLD_FAIL"):
        raise SystemExit(23)
else:
    raise SystemExit(f"unexpected script: {script_name}")
""",
        encoding="utf-8",
    )
    shim = tmp_path / "fake-python.cmd"
    shim.write_text(
        f'@"{sys.executable}" "{dispatcher}" %*\n',
        encoding="utf-8",
    )
    return shim, tmp_path / "calls.jsonl"


def _run_soak(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
    *,
    extra_env: dict[str, str] | None = None,
    include_seeds: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]], dict[str, str], Path]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is required for soak orchestration tests")

    shim, call_log = fake_python
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "game_stale.json").write_text(
        json.dumps({"game_id": "stale-must-not-be-read"}),
        encoding="utf-8",
    )
    state_path = tmp_path / "state.json"
    launch_cwd = tmp_path / "caller-cwd"
    launch_cwd.mkdir()
    wrapper = tmp_path / "invoke-soak.ps1"
    seeds_argument = " -Seeds (714001..714010)" if include_seeds else ""
    wrapper.write_text(
        f"""$ErrorActionPreference = 'Stop'
$originalLocation = (Get-Location).Path
$env:WEREWOLF_GAME_LOG_PATH = 'sentinel-log-path'
$exitCode = 0
try {{
    & '{SOAK_SCRIPT}'{seeds_argument} -PythonCommand '{shim}' -ArtifactRoot '{artifact_root}'
}}
catch {{
    Write-Error $_
    $exitCode = 1
}}
finally {{
    @{{
        cwd = (Get-Location).Path
        original_cwd = $originalLocation
        game_log_path = $env:WEREWOLF_GAME_LOG_PATH
    }} | ConvertTo-Json | Set-Content -LiteralPath '{state_path}' -Encoding utf8
}}
exit $exitCode
""",
        encoding="utf-8-sig",
    )
    env = os.environ.copy()
    env["FAKE_PYTHON_CALL_LOG"] = str(call_log)
    env.update(extra_env or {})
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
        ],
        cwd=launch_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    calls = [
        json.loads(line)
        for line in call_log.read_text(encoding="utf-8").splitlines()
    ] if call_log.exists() else []
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    return result, calls, state, artifact_root


def _option_value(call: dict[str, object], option: str) -> str:
    args = call["args"]
    assert isinstance(args, list)
    return str(args[args.index(option) + 1])


def test_soak_runs_exactly_ten_isolated_games_then_analyzer_and_evaluator(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, _ = _run_soak(tmp_path, fake_python)

    assert result.returncode == 0, result.stderr
    assert [call["script"] for call in calls] == [
        *("run_real_game.py" for _ in range(10)),
        "analyze_recent_balance.py",
        "evaluate_audit_closure_thresholds.py",
    ]
    runners = calls[:10]
    assert [_option_value(call, "--seed") for call in runners] == [
        str(seed) for seed in range(714001, 714011)
    ]
    assert len({_option_value(call, "--game-id") for call in runners}) == 10
    assert len({_option_value(call, "--output-dir") for call in runners}) == 10
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"


def test_soak_does_not_pass_removed_agent_timeout_options(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, _, _ = _run_soak(tmp_path, fake_python)

    assert result.returncode == 0, result.stderr
    runner_calls = [call for call in calls if call["script"] == "run_real_game.py"]
    assert runner_calls
    assert all("--timeout" not in call["args"] for call in runner_calls)
    assert all("--no-timeout" not in call["args"] for call in runner_calls)
    assert "TimeoutSeconds" not in SOAK_SCRIPT.read_text(encoding="utf-8")


def test_soak_requires_ten_explicit_seeds(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, _ = _run_soak(
        tmp_path,
        fake_python,
        include_seeds=False,
    )

    assert result.returncode != 0
    assert calls == []
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"


def test_soak_retains_aborted_launch_without_replacement_and_reports_counts(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, artifact_root = _run_soak(
        tmp_path,
        fake_python,
        extra_env={"FAKE_ABORT_AT": "3"},
    )

    assert result.returncode == 0, result.stderr
    assert [call["script"] for call in calls[:10]] == ["run_real_game.py"] * 10
    manifests = list(artifact_root.rglob("audit-closure-soak-manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8-sig"))
    assert manifest["launch_count"] == 10
    assert manifest["finished_count"] == 9
    assert manifest["aborted_count"] == 1
    assert [row["status"] for row in manifest["games"]].count("aborted") == 1
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"


def test_soak_analyzes_only_new_run_scoped_game_artifacts(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, _, artifact_root = _run_soak(tmp_path, fake_python)

    assert result.returncode == 0, result.stderr
    analyzer = next(
        call for call in calls if call["script"] == "analyze_recent_balance.py"
    )
    analyzer_args = analyzer["args"]
    assert isinstance(analyzer_args, list)
    assert len(analyzer_args) == 10
    assert all("game_stale.json" not in str(path) for path in analyzer_args)
    run_dirs = [path for path in artifact_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1


def test_soak_stops_immediately_when_a_runner_fails_and_restores_state(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, _ = _run_soak(
        tmp_path,
        fake_python,
        extra_env={"FAKE_RUNNER_FAIL_AT": "3"},
    )

    assert result.returncode != 0
    assert [call["script"] for call in calls] == ["run_real_game.py"] * 3
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"


def test_soak_rejects_runner_json_outside_seed_output_directory(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, _ = _run_soak(
        tmp_path,
        fake_python,
        extra_env={"FAKE_RUNNER_PATH_MODE": "outside"},
    )

    assert result.returncode != 0
    assert [call["script"] for call in calls] == ["run_real_game.py"]
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"


def test_soak_rejects_runner_json_through_symlink_escape(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, _ = _run_soak(
        tmp_path,
        fake_python,
        extra_env={"FAKE_RUNNER_PATH_MODE": "symlink_escape"},
    )

    assert result.returncode != 0
    assert [call["script"] for call in calls] == ["run_real_game.py"]
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"


def test_soak_preserves_threshold_diagnostics_when_evaluation_fails(
    tmp_path: Path,
    fake_python: tuple[Path, Path],
) -> None:
    result, calls, state, artifact_root = _run_soak(
        tmp_path,
        fake_python,
        extra_env={"FAKE_THRESHOLD_FAIL": "1"},
    )

    assert result.returncode != 0
    assert [call["script"] for call in calls[-2:]] == [
        "analyze_recent_balance.py",
        "evaluate_audit_closure_thresholds.py",
    ]
    threshold_files = list(artifact_root.rglob("audit-closure-thresholds.json"))
    report_files = list(artifact_root.rglob("audit-closure-report.json"))
    assert len(threshold_files) == 1
    assert json.loads(threshold_files[0].read_text(encoding="utf-8")) == {
        "overall_pass": False
    }
    assert len(report_files) == 1
    assert state["cwd"] == state["original_cwd"]
    assert state["game_log_path"] == "sentinel-log-path"
