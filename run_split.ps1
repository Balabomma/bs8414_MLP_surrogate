# Full train -> evaluate -> validate -> killer-excluded chain for ONE split ratio.
#
# The split is selected by the MLP_SPLIT environment variable, which config.py
# reads; with it unset everything is the default 70/15/15 KAN-ablation contract.
# Each ratio gets its own checkpoint directory (never re-pointed) and its own
# outputs/split_<ratio>/ tree.
#
#   .\run_split.ps1 70_15_15
#   .\run_split.ps1 80_10_10
#   .\run_split.ps1 -Split 70_15_15 -Suffix _r2     # replicate 2
#
# Replicates are IDENTICAL reruns — same code, same split, same seeds. The draw
# variation comes from cudnn.benchmark non-determinism, which is exactly how the
# KAN reference population was built (bestofn_v9_driver.py). Each replicate gets
# its own directory; nothing is ever re-pointed.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("70_15_15", "80_10_10")]
    [string]$Split,

    [ValidatePattern("^(_r[0-9]+)?$")]
    [string]$Suffix = "",

    # Optional experiment tag inserted before the replicate suffix, e.g.
    # -Tag _grouped -> models_mlp_70_15_15_grouped_r2. Used to keep the
    # configuration-grouped-split runs in their own directories so the
    # earlier CHID-split checkpoints are never re-pointed.
    [ValidatePattern("^(_[A-Za-z0-9]+)?$")]
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot
$py = Join-Path $proj "venv\Scripts\python.exe"
$modelDir = "models_mlp_${Split}${Tag}${Suffix}"
$outDir = Join-Path $proj "outputs\split_${Split}${Tag}${Suffix}"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$env:MLP_SPLIT = $Split

if (Test-Path (Join-Path $proj "$modelDir\best_model.pt")) {
    throw "$modelDir already holds a trained ensemble - refusing to overwrite it. " +
          "Pick a fresh -Suffix."
}

Write-Output "=== split $Split -> $modelDir ==="
Write-Output "start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

& $py -u (Join-Path $proj "train.py") --model-dir $modelDir `
    1> (Join-Path $outDir "train.log") 2> (Join-Path $outDir "train.err.log")
if ($LASTEXITCODE -ne 0) { throw "train.py failed for $Split (exit $LASTEXITCODE)" }
Write-Output "train done $(Get-Date -Format 'HH:mm:ss')"

& $py -u (Join-Path $proj "evaluate.py") --model-dir $modelDir `
    1> (Join-Path $outDir "evaluate.log") 2> (Join-Path $outDir "evaluate.err.log")
if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $Split (exit $LASTEXITCODE)" }
Write-Output "evaluate done $(Get-Date -Format 'HH:mm:ss')"

& $py -u (Join-Path $proj "validate_physics.py") --model-dir $modelDir `
    1> (Join-Path $outDir "validate_physics.log") 2> (Join-Path $outDir "validate_physics.err.log")
if ($LASTEXITCODE -ne 0) { throw "validate_physics.py failed for $Split (exit $LASTEXITCODE)" }
Write-Output "validate done $(Get-Date -Format 'HH:mm:ss')"

& $py -u (Join-Path $proj "killer_excluded.py") --model-dir $modelDir `
    1> (Join-Path $outDir "killer_excluded.log") 2> (Join-Path $outDir "killer_excluded.err.log")
if ($LASTEXITCODE -ne 0) { throw "killer_excluded.py failed for $Split (exit $LASTEXITCODE)" }
Write-Output "killer-excluded done $(Get-Date -Format 'HH:mm:ss')"

# Mirror the frozen-contract artifacts (CSVs, figures, summary.txt) next to the logs
$src = Join-Path $proj "$modelDir\outputs"
$dst = Join-Path $outDir "metrics"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force

Write-Output "=== split $Split complete $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "  checkpoints -> $modelDir"
Write-Output "  outputs     -> $outDir"
