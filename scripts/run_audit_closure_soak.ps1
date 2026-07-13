<#
.SYNOPSIS
运行固定种子的真实模型验收浸泡测试，并只分析本次运行产生的对局。

.DESCRIPTION
默认运行 10 局，种子为 713001..713010。每局使用独立的
WEREWOLF_GAME_LOG_PATH，并从 run_real_game.py 的明确输出读取对应游戏 JSON；
不会扫描或混入仓库中的历史 game_*.json。任一游戏或分析命令失败时立即退出。

硬门槛（详见审查闭环设计文档）：完局率 100%；胜负后游戏内模型调用、弱计划击杀、
无目标证据 fallback 击杀、fallback_disabled、不支持公开事实、聚合差异均为 0；
狼队计划终态 fallback <10%，schema fallback <5%；Persona、关键任务推理覆盖、
语义目标保持/无新增 claim、可能世界唯一性与 evidence ref、神职伤害证据完整性均 100%；
可修复语义问题成功率 >=95%。本脚本不会自动发起超过参数指定数量的付费对局。

.EXAMPLE
pwsh -File scripts/run_audit_closure_soak.ps1
#>

[CmdletBinding()]
param(
    [int]$GameCount = 10,
    [int]$StartSeed = 713001,
    [int]$MaxSteps = 500,
    [double]$TimeoutSeconds = 120,
    [int]$DelayMilliseconds = 0
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$outputDir = Join-Path $root "artifacts/audit_closure_soak/$runId"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$gameFiles = [System.Collections.Generic.List[string]]::new()
for ($offset = 0; $offset -lt $GameCount; $offset++) {
    $seed = $StartSeed + $offset
    $env:WEREWOLF_GAME_LOG_PATH = Join-Path $outputDir "seed-$seed.stdout.log"
    $runOutput = & python scripts/run_real_game.py `
        --seed $seed `
        --max-steps $MaxSteps `
        --timeout $TimeoutSeconds `
        --delay $DelayMilliseconds 2>&1
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

if ($gameFiles.Count -ne $GameCount) {
    throw "Expected $GameCount game JSON files, collected $($gameFiles.Count)"
}

$reportPath = Join-Path $outputDir 'audit-closure-report.json'
$reportJson = & python scripts/analyze_recent_balance.py @gameFiles
if ($LASTEXITCODE -ne 0) {
    throw "Audit analyzer failed with exit code $LASTEXITCODE"
}
$reportJson | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Host "Audit closure report: $reportPath"
