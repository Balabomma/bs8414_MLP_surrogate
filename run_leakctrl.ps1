# Leakage control: does having mesh siblings in training help, once the number
# of distinct training configurations is held fixed?
#
# The first run (models_mlp_leakctrl, seed 42, unswapped) returned leaked 0.799
# against clean 0.876 -- no benefit, and the point estimate in the wrong
# direction. That single run scores 4 simulations per arm against a
# configuration-level SD of about 0.20, so it cannot resolve the effect. This
# script adds the two things that make the result publishable either way:
#
#   replicates       seeds 1337 and 2024, so the gap has a distribution
#   swapped design   leak and clean configurations exchanged, so the result
#                    cannot be an artefact of which draw landed in which arm
#
# Each run trains ONCE and is scored TWICE (clean arm, leaked arm). Training and
# validation are byte-identical between arms by construction, so the difference
# between the two scores is the leakage effect with nothing else moving.
#
# Sequential. Resumable: a run whose best_model.pt exists is re-scored but not
# retrained. Never wrap in Start-Job.

$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot
$py = Join-Path $proj "venv\Scripts\python.exe"
$env:PYTHONPATH = $proj
$env:MLP_SPLIT = "70_15_15"

# name, seed base, swapped
$runs = @(
    @{ name = "models_mlp_leakctrl";           seed = "42";   swap = "0" },   # already trained
    @{ name = "models_mlp_leakctrl_seed1337";  seed = "1337"; swap = "0" },
    @{ name = "models_mlp_leakctrl_seed2024";  seed = "2024"; swap = "0" },
    @{ name = "models_mlp_leakctrl_swapped";   seed = "42";   swap = "1" },
    @{ name = "models_mlp_leakctrl_swap1337";  seed = "1337"; swap = "1" },
    @{ name = "models_mlp_leakctrl_swap2024";  seed = "2024"; swap = "1" }
)

Write-Output "leakage control start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

foreach ($r in $runs) {
    $name = $r.name
    $out = Join-Path $proj "outputs\campaign\$name"
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    $env:MLP_SEED_BASE = $r.seed
    $env:MLP_LEAKCTRL_SWAP = $r.swap

    if (Test-Path (Join-Path $proj "$name\best_model.pt")) {
        Write-Host "  [skip-train] $name  (already trained)"
    }
    else {
        Write-Host "  [train] $name  seed=$($r.seed) swap=$($r.swap)  $(Get-Date -Format 'HH:mm:ss')"
        $env:MLP_CAMPAIGN = "leakctrl_clean"
        & $py -u (Join-Path $proj "train.py") --model-dir $name `
            1> (Join-Path $out "train.log") 2> (Join-Path $out "train.err.log")
        if ($LASTEXITCODE -ne 0) { throw "train failed for $name" }
    }

    foreach ($arm in @("clean", "leaked")) {
        $env:MLP_CAMPAIGN = "leakctrl_$arm"
        & $py -u (Join-Path $proj "evaluate.py") --model-dir $name `
            1> (Join-Path $out "eval_$arm.log") 2> (Join-Path $out "eval_$arm.err.log")
        if ($LASTEXITCODE -ne 0) { throw "evaluate ($arm) failed for $name" }
        Copy-Item (Join-Path $proj "$name\outputs\overall_metrics.csv") `
                  (Join-Path $out "metrics_$arm.csv") -Force
    }
    Write-Host "  [done ] $name  $(Get-Date -Format 'HH:mm:ss')"
}

Remove-Item Env:MLP_SEED_BASE, Env:MLP_LEAKCTRL_SWAP, Env:MLP_CAMPAIGN -ErrorAction SilentlyContinue
Write-Output "leakage control end   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "aggregate with: python collect_leakctrl.py"
