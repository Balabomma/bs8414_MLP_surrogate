# Population comparison across all replicates of both split ratios.
#
# Run this only after every replicate chain has finished. The chains themselves are
# launched individually:
#
#   .\run_split.ps1 -Split 70_15_15 -Tag _grouped -Suffix _r2
#   .\run_split.ps1 -Split 80_10_10 -Tag _grouped -Suffix _r2
#
# NOTE: do NOT wrap those in Start-Job. Job sessions are children of the launching
# PowerShell process and are terminated with it — that killed replicate-2 mid-run
# on 2026-08-11 (70/15/15 at candidate 8/12, 80/10/10 at 6/12, ~60 min lost each,
# nothing recoverable because train.py only checkpoints after all 12 candidates).
# Launch each chain as its own top-level background process instead.
#
#   .\run_population_comparison.ps1                 # grouped headline + CHID appendix
#   .\run_population_comparison.ps1 -Tag ""         # compare the legacy CHID line
param(
    [ValidatePattern("^(_[A-Za-z0-9]+)?$")]
    [string]$Tag = "_grouped"
)

$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot
$py = Join-Path $proj "venv\Scripts\python.exe"
$splits = @("70_15_15", "80_10_10")
$reps = @("", "_r2", "_r3")
$mode = if ($Tag -eq "_grouped") { "grouped" } else { "chid" }

# Only compare replicates that actually produced an ensemble, and print the roster
# explicitly so a missing chain can never be silently averaged away.
$present = @{}
foreach ($split in $splits) {
    $dirs = @()
    foreach ($rep in $reps) {
        $name = "models_mlp_${split}${Tag}${rep}"
        if (Test-Path (Join-Path $proj "$name\best_model.pt")) { $dirs += $name }
        else { Write-Output "  [missing] $name - no best_model.pt, excluded" }
    }
    $present[$split] = $dirs
    Write-Output "  $split$Tag : $($dirs.Count) replicate(s) - $($dirs -join ', ')"
}
if ($present["70_15_15"].Count -eq 0 -or $present["80_10_10"].Count -eq 0) {
    throw "one side has no trained replicates - nothing to compare"
}

$cmpDir = Join-Path $proj "outputs\comparison"
New-Item -ItemType Directory -Force -Path $cmpDir | Out-Null
$report = Join-Path $cmpDir "split_comparison$Tag.md"

& $py -u (Join-Path $proj "compare_splits.py") `
    --dirs-a $present["70_15_15"] --dirs-b $present["80_10_10"] `
    --split-mode $mode --out $report `
    1> (Join-Path $cmpDir "compare_splits$Tag.log") 2> (Join-Path $cmpDir "compare_splits$Tag.err.log")
if ($LASTEXITCODE -ne 0) { throw "compare_splits.py failed (exit $LASTEXITCODE)" }

# Give every replicate's output folder its own copy, so each is self-contained.
foreach ($split in $splits) {
    foreach ($name in $present[$split]) {
        $suffix = $name -replace "^models_mlp_${split}${Tag}", ""
        $dest = Join-Path $proj "outputs\split_${split}${Tag}${suffix}"
        if (Test-Path $dest) {
            Copy-Item $report (Join-Path $dest "split_comparison.md") -Force
        }
    }
}

Write-Output "comparison complete $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Get-Content (Join-Path $cmpDir "compare_splits$Tag.log") -Tail 50
