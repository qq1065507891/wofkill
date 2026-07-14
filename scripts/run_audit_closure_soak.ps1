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
    [int]$DelayMilliseconds = 0,
    [string]$PythonCommand = 'python',
    [string]$ArtifactRoot = ''
)

$ErrorActionPreference = 'Stop'

function Resolve-SeedGameArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$ReportedPath,
        [Parameter(Mandatory = $true)][string]$SeedOutputDir
    )

    $seedRoot = [System.IO.Path]::GetFullPath($SeedOutputDir).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $candidate = [System.IO.Path]::GetFullPath($ReportedPath)
    $seedPrefix = $seedRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith(
        $seedPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Game JSON must stay under its seed output directory: $candidate"
    }

    $current = Get-Item -LiteralPath $candidate -Force
    while ($null -ne $current) {
        if (($current.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Game JSON path must not traverse a symbolic link: $candidate"
        }
        if ($current.FullName.Equals(
            $seedRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            break
        }
        $current = if ($current -is [System.IO.DirectoryInfo]) {
            $current.Parent
        }
        else {
            $current.Directory
        }
    }
    if ($null -eq $current) {
        throw "Game JSON parent chain did not reach its seed output directory: $candidate"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

if ($Seeds.Count -ne 10) {
    throw "Audit closure soak requires exactly 10 seeds; received $($Seeds.Count)"
}
if (@($Seeds | Select-Object -Unique).Count -ne 10) {
    throw "Audit closure soak requires 10 unique seeds"
}

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runnerScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'run_real_game.py'
)).Path
$analyzerScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'analyze_recent_balance.py'
)).Path
$thresholdEvaluatorScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'evaluate_audit_closure_thresholds.py'
)).Path
$runId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$([guid]::NewGuid().ToString('N'))"
$artifactBase = if ($ArtifactRoot) {
    [System.IO.Path]::GetFullPath($ArtifactRoot)
}
else {
    Join-Path $root 'artifacts/audit_closure_soak'
}
$outputDir = Join-Path $artifactBase $runId
if (Test-Path -LiteralPath $outputDir) {
    throw "Refusing to reuse audit closure artifact directory: $outputDir"
}
New-Item -ItemType Directory -Path $outputDir | Out-Null

$previousGameLogPath = $env:WEREWOLF_GAME_LOG_PATH
$gameFiles = [System.Collections.Generic.List[string]]::new()
$gameIds = [System.Collections.Generic.HashSet[string]]::new()
Push-Location -LiteralPath $root
try {
    foreach ($seed in $Seeds) {
        $gameId = "audit-$runId-seed-$seed"
        if (-not $gameIds.Add($gameId)) {
            throw "Refusing duplicate run-scoped game ID: $gameId"
        }
        $gameOutputDir = Join-Path $outputDir "seed-$seed"
        New-Item -ItemType Directory -Path $gameOutputDir | Out-Null
        $env:WEREWOLF_GAME_LOG_PATH = Join-Path $gameOutputDir 'runner.stdout.log'
        $runOutput = & $PythonCommand $runnerScript `
            --seed $seed `
            --game-id $gameId `
            --max-steps $MaxSteps `
            --timeout $TimeoutSeconds `
            --delay $DelayMilliseconds `
            --output-dir $gameOutputDir 2>&1
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
        $reportedGamePath = $Matches[1].Trim()
        $gamePath = Resolve-SeedGameArtifact `
            -ReportedPath $reportedGamePath `
            -SeedOutputDir $gameOutputDir
        if ($gameFiles.Contains($gamePath)) {
            throw "Refusing duplicate game JSON path: $gamePath"
        }
        $gameData = Get-Content -LiteralPath $gamePath -Raw | ConvertFrom-Json
        if ($gameData.game_id -ne $gameId) {
            throw "Game JSON ID mismatch: expected $gameId, got $($gameData.game_id)"
        }
        $gameFiles.Add($gamePath)
    }

    if ($gameFiles.Count -ne 10) {
        throw "Expected 10 game JSON files, collected $($gameFiles.Count)"
    }

    $reportPath = Join-Path $outputDir 'audit-closure-report.json'
    $reportJson = & $PythonCommand $analyzerScript @gameFiles
    if ($LASTEXITCODE -ne 0) {
        throw "Audit analyzer failed with exit code $LASTEXITCODE"
    }
    $reportJson | Set-Content -LiteralPath $reportPath -Encoding utf8
    Write-Host "Audit closure report: $reportPath"

    $thresholdPath = Join-Path $outputDir 'audit-closure-thresholds.json'
    & $PythonCommand $thresholdEvaluatorScript $reportPath $thresholdPath
    $thresholdExitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $thresholdPath)) {
        throw "Threshold evaluator did not write $thresholdPath"
    }
    Write-Host "Audit closure thresholds: $thresholdPath"
    if ($thresholdExitCode -ne 0) {
        throw "Audit closure hard thresholds failed; inspect $thresholdPath"
    }
}
finally {
    Pop-Location
    $env:WEREWOLF_GAME_LOG_PATH = $previousGameLogPath
}
