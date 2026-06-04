[CmdletBinding()]
param(
    [string[]]$Refs = @("68c4c62", "WORKTREE"),
    [string]$BaselineRef = "68c4c62",
    [int]$Runs = 3,
    [int]$Seconds = 20,
    [double]$AllowedRegression = 0.05,
    [switch]$SkipSecondary,
    [switch]$ReportOnly,
    [string]$WorktreeRoot = (Join-Path ([System.IO.Path]::GetTempPath()) "neo-quad-stress-perf-worktrees"),
    [string]$OutputCsv = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Refs = @(
    foreach ($ref in $Refs) {
        foreach ($part in ($ref -split ",")) {
            $trimmed = $part.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                $trimmed
            }
        }
    }
)

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        Write-Host "> $FilePath $($Arguments -join ' ')"
        $ErrorActionPreference = "Continue"
        $output = & $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }

    if ($exitCode -ne 0) {
        $tail = ($output | Select-Object -Last 80) -join "`n"
        throw "Command failed with exit code $exitCode`: $FilePath $($Arguments -join ' ')`n$tail"
    }

    return @($output)
}

function Get-RepoRoot {
    $root = (& git rev-parse --show-toplevel 2>$null).Trim()
    if (-not $root) {
        throw "Run this script from inside the Neo git repository."
    }
    return $root
}

function Resolve-Commit {
    param(
        [string]$RepoRoot,
        [string]$Ref
    )

    if ($Ref -in @("WORKTREE", ".", "CURRENT")) {
        return (& git -C $RepoRoot rev-parse --verify "HEAD^{commit}").Trim()
    }

    return (& git -C $RepoRoot rev-parse --verify "$Ref^{commit}").Trim()
}

function Get-SafeLabel {
    param([string]$Ref, [string]$Commit)

    if ($Ref -in @("WORKTREE", ".", "CURRENT")) {
        return "WORKTREE-$($Commit.Substring(0, 7))"
    }

    $label = $Ref -replace '[^A-Za-z0-9._-]', '_'
    if ([string]::IsNullOrWhiteSpace($label)) {
        $label = $Commit.Substring(0, 7)
    }
    return $label
}

function Ensure-PerfWorktree {
    param(
        [string]$RepoRoot,
        [string]$Ref,
        [string]$Commit
    )

    if ($Ref -in @("WORKTREE", ".", "CURRENT")) {
        return $RepoRoot
    }

    New-Item -ItemType Directory -Force -Path $WorktreeRoot | Out-Null
    $label = Get-SafeLabel -Ref $Ref -Commit $Commit
    $path = Join-Path $WorktreeRoot $label

    if (Test-Path $path) {
        $existingCommit = (& git -C $path rev-parse --verify "HEAD^{commit}").Trim()
        if ($existingCommit -ne $Commit) {
            throw "Existing worktree $path is at $existingCommit, expected $Commit. Remove it or choose a different -WorktreeRoot."
        }
    } else {
        Invoke-Captured -FilePath "git" -Arguments @("worktree", "add", "--detach", $path, $Commit) -WorkingDirectory $RepoRoot | Out-Null
    }

    $dirty = @(& git -C $path status --porcelain)
    if ($dirty.Count -gt 0) {
        throw "Worktree $path is dirty. Clean it before benchmarking.`n$($dirty -join "`n")"
    }

    return $path
}

function Get-Median {
    param([double[]]$Values)

    if ($Values.Count -eq 0) {
        throw "Cannot compute median of an empty set."
    }

    $sorted = @($Values | Sort-Object)
    $mid = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$mid]
    }

    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

function ConvertTo-Double {
    param([string]$Value)
    return [double]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-RunSummary {
    param(
        [string[]]$Output,
        [string]$Ref,
        [string]$Commit,
        [string]$Scenario,
        [int]$Run
    )

    $pattern = [regex]'kernel_fps\s+(?<kernel>[0-9]+(?:\.[0-9]+)?)\s+\|.*?present_fps\s+(?<presentFps>[0-9]+(?:\.[0-9]+)?)\s+\|.*?gpu_copy\s+(?<gpuCopy>[0-9]+(?:\.[0-9]+)?)\s+us\s+\|\s+swap\s+(?<swap>[0-9]+(?:\.[0-9]+)?)\s+us\s+\|\s+present\s+(?<presentUs>[0-9]+(?:\.[0-9]+)?)\s+us'
    $samples = foreach ($line in $Output) {
        $match = $pattern.Match($line)
        if ($match.Success) {
            [pscustomobject]@{
                KernelFps  = ConvertTo-Double $match.Groups["kernel"].Value
                PresentFps = ConvertTo-Double $match.Groups["presentFps"].Value
                GpuCopyUs  = ConvertTo-Double $match.Groups["gpuCopy"].Value
                SwapUs     = ConvertTo-Double $match.Groups["swap"].Value
                PresentUs  = ConvertTo-Double $match.Groups["presentUs"].Value
            }
        }
    }

    $samples = @($samples)
    if ($samples.Count -eq 0) {
        $tail = ($Output | Select-Object -Last 80) -join "`n"
        throw "No perf metric lines were parsed for ref $Ref scenario $Scenario run $Run.`n$tail"
    }

    return [pscustomobject]@{
        Scenario   = $Scenario
        Ref        = $Ref
        Commit     = $Commit.Substring(0, 12)
        Run        = $Run
        Samples    = $samples.Count
        KernelFps  = Get-Median @($samples.KernelFps)
        PresentFps = Get-Median @($samples.PresentFps)
        GpuCopyUs  = Get-Median @($samples.GpuCopyUs)
        SwapUs     = Get-Median @($samples.SwapUs)
        PresentUs  = Get-Median @($samples.PresentUs)
    }
}

function Get-ScenarioSummaries {
    param([object[]]$Rows)

    $groups = $Rows | Group-Object Scenario, Ref
    foreach ($group in $groups) {
        $items = @($group.Group)
        [pscustomobject]@{
            Scenario   = $items[0].Scenario
            Ref        = $items[0].Ref
            Commit     = $items[0].Commit
            Runs       = $items.Count
            Samples    = ($items | Measure-Object -Property Samples -Sum).Sum
            KernelFps  = Get-Median @($items.KernelFps)
            PresentFps = Get-Median @($items.PresentFps)
            GpuCopyUs  = Get-Median @($items.GpuCopyUs)
            SwapUs     = Get-Median @($items.SwapUs)
            PresentUs  = Get-Median @($items.PresentUs)
        }
    }
}

$repoRoot = Get-RepoRoot
$baselineCommit = Resolve-Commit -RepoRoot $repoRoot -Ref $BaselineRef
$refSpecs = @()

foreach ($ref in @($BaselineRef) + $Refs) {
    $existingRefs = @($refSpecs | ForEach-Object { $_.Ref })
    if ($existingRefs -contains $ref) {
        continue
    }
    $commit = Resolve-Commit -RepoRoot $repoRoot -Ref $ref
    $path = Ensure-PerfWorktree -RepoRoot $repoRoot -Ref $ref -Commit $commit
    $refSpecs += [pscustomobject]@{
        Ref    = $ref
        Commit = $commit
        Path   = $path
    }
}

$scenarios = @(
    [pscustomobject]@{
        Name = "primary"
        Args = @(
            "--seconds", "$Seconds",
            "--render-policy", "force-render",
            "--no-hot-reload"
        )
    }
)

if (-not $SkipSecondary) {
    $scenarios += [pscustomobject]@{
        Name = "macrocell-sparse"
        Args = @(
            "--seconds", "$Seconds",
            "--draw-backend", "cuda-tiled",
            "--instance-stress-variant", "macrocell",
            "--instance-materials", "sparse-texture",
            "--sparse-feedback", "off",
            "--gpu-preset", "auto",
            "--render-policy", "force-render",
            "--kernel-target-fps", "0",
            "--no-hot-reload"
        )
    }
}

$builtSpecs = @()
foreach ($spec in $refSpecs) {
    Write-Host ""
    Write-Host "=== Building $($spec.Ref) ($($spec.Commit.Substring(0, 12))) ==="
    Invoke-Captured -FilePath "cargo" -Arguments @("build", "-p", "neo-quad-stress-3d", "--release") -WorkingDirectory $spec.Path | Out-Null

    $exe = Join-Path $spec.Path "target\release\neo-quad-stress-3d.exe"
    if (-not (Test-Path $exe)) {
        $exe = Join-Path $spec.Path "target\release\neo-quad-stress-3d"
    }
    if (-not (Test-Path $exe)) {
        throw "Built executable not found under $($spec.Path)\target\release."
    }

    $builtSpecs += [pscustomobject]@{
        Ref    = $spec.Ref
        Commit = $spec.Commit
        Path   = $spec.Path
        Exe    = $exe
    }
}

$rows = @()
foreach ($scenario in $scenarios) {
    for ($run = 1; $run -le $Runs; $run++) {
        foreach ($spec in $builtSpecs) {
            Write-Host ""
            Write-Host "=== $($spec.Ref) :: $($scenario.Name) :: run $run/$Runs ==="
            $output = Invoke-Captured -FilePath $spec.Exe -Arguments ([string[]]$scenario.Args) -WorkingDirectory $spec.Path
            $rows += Get-RunSummary -Output $output -Ref $spec.Ref -Commit $spec.Commit -Scenario $scenario.Name -Run $run
        }
    }
}

if ([string]::IsNullOrWhiteSpace($OutputCsv)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputCsv = Join-Path $repoRoot "target\perf\quad_stress_perf_$timestamp.csv"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputCsv) | Out-Null
$rows | Export-Csv -NoTypeInformation -Path $OutputCsv

$summaries = @(Get-ScenarioSummaries -Rows $rows)
$gate = 1.0 - $AllowedRegression
$failures = @()
$comparisonRows = @()

foreach ($scenario in @($summaries.Scenario | Sort-Object -Unique)) {
    $baseline = $summaries | Where-Object {
        $_.Scenario -eq $scenario -and $_.Ref -eq $BaselineRef
    } | Select-Object -First 1
    if (-not $baseline) {
        throw "Missing baseline summary for scenario $scenario ref $BaselineRef."
    }

    foreach ($summary in $summaries | Where-Object { $_.Scenario -eq $scenario }) {
        $ratio = $summary.KernelFps / $baseline.KernelFps
        $status = if ($summary.Ref -eq $BaselineRef -or $ratio -ge $gate) { "PASS" } else { "FAIL" }
        $comparison = [pscustomobject]@{
            Scenario       = $summary.Scenario
            Ref            = $summary.Ref
            Commit         = $summary.Commit
            Runs           = $summary.Runs
            Samples        = $summary.Samples
            KernelFps      = [Math]::Round($summary.KernelFps, 1)
            BaselineFps    = [Math]::Round($baseline.KernelFps, 1)
            Ratio          = [Math]::Round($ratio, 4)
            PresentFps     = [Math]::Round($summary.PresentFps, 1)
            GpuCopyUs      = [Math]::Round($summary.GpuCopyUs, 1)
            SwapUs         = [Math]::Round($summary.SwapUs, 1)
            PresentUs      = [Math]::Round($summary.PresentUs, 1)
            Status         = $status
        }
        $comparisonRows += $comparison
        if ($status -eq "FAIL") {
            $failures += $comparison
        }
    }
}

Write-Host ""
Write-Host "=== Quad stress perf summary ==="
$comparisonRows | Sort-Object Scenario, Ref | Format-Table -AutoSize
Write-Host "Raw run CSV: $OutputCsv"

if ($failures.Count -gt 0) {
    $message = "Perf gate failed: $($failures.Count) scenario/ref result(s) fell below $([Math]::Round($gate * 100.0, 1))% of baseline $BaselineRef."
    if ($ReportOnly) {
        Write-Warning $message
    } else {
        throw $message
    }
}
