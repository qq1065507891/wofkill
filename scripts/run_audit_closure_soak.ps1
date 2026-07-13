<#
.SYNOPSIS
运行固定十个种子的真实模型验收浸泡测试，并隔离本次全部产物。

.DESCRIPTION
默认种子为 713001..713010；自定义 Seeds 也必须恰好包含十项。脚本从自身目录
推导仓库根和 Python 脚本绝对路径，因此可从任意当前目录调用。每局标准输出和游戏
JSON 均写入本次时间戳 artifact 目录，任一命令失败立即停止。

硬门槛详见审查闭环设计：完局率、Persona、关键任务推理、语义修复、可能世界和神职
证据均须满足设计值；终态 fallback <10%，schema fallback <5%；危险击杀、胜负后调用、
fallback_disabled、不支持事实和聚合差异均须为零。此脚本不会自行扩大付费样本数。

.EXAMPLE
pwsh -File scripts/run_audit_closure_soak.ps1

.EXAMPLE
pwsh -File scripts/run_audit_closure_soak.ps1 -Seeds (800001..800010)
#>

[CmdletBinding()]
param(
    [int[]]$Seeds = (713001..713010),
    [int]$MaxSteps = 500,
    [double]$TimeoutSeconds = 120,
    [int]$DelayMilliseconds = 0
)

$ErrorActionPreference = 'Stop'
if ($Seeds.Count -ne 10) {
    throw "Audit closure soak requires exactly 10 seeds; received $($Seeds.Count)"
}

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runnerScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'run_real_game.py'
)).Path
$analyzerScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'analyze_recent_balance.py'
)).Path
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$outputDir = Join-Path $root "artifacts/audit_closure_soak/$runId"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$previousGameLogPath = $env:WEREWOLF_GAME_LOG_PATH
$gameFiles = [System.Collections.Generic.List[string]]::new()
Push-Location -LiteralPath $root
try {
    foreach ($seed in $Seeds) {
        $env:WEREWOLF_GAME_LOG_PATH = Join-Path $outputDir "seed-$seed.stdout.log"
        $runOutput = & python $runnerScript `
            --seed $seed `
            --max-steps $MaxSteps `
            --timeout $TimeoutSeconds `
            --delay $DelayMilliseconds `
            --output-dir $outputDir 2>&1
        $runExitCode = $LASTEXITCODE
        $runOutput | ForEach-Object { Write-Host $_ }
        if ($runExitCode -ne 0) {
            throw "Real game failed for seed $seed with exit code $runExitCode"
        }

        $gameLogLine = $runOutput | Where-Object {
            $_ -match '^\s*Game log:\s*(.+\.json)\s*$'
        } | Select-Object -Last 1
        if (-not $gameLogLine -or $gameLogLine -notmatch '^\s*Game log:\s*(.+\.json)\s*$') {
            throw "Real game for seed $seed did not report its JSON path"
        }
        $gamePath = (Resolve-Path -LiteralPath $Matches[1].Trim()).Path
        $gameFiles.Add($gamePath)
    }

    if ($gameFiles.Count -ne 10) {
        throw "Expected 10 game JSON files, collected $($gameFiles.Count)"
    }

    $reportPath = Join-Path $outputDir 'audit-closure-report.json'
    $reportJson = & python $analyzerScript @gameFiles
    if ($LASTEXITCODE -ne 0) {
        throw "Audit analyzer failed with exit code $LASTEXITCODE"
    }
    $reportJson | Set-Content -LiteralPath $reportPath -Encoding utf8
    Write-Host "Audit closure report: $reportPath"
}
finally {
    Pop-Location
    $env:WEREWOLF_GAME_LOG_PATH = $previousGameLogPath
}
