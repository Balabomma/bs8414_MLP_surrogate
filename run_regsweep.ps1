# Regularisation sweep: close the train/test generalisation gap.
#
# Diagnosis this is built on. Across three grouped replicates the model scores
# train R2 = 0.972, valid R2 = 0.890, test R2 = 0.718, and test MAPE = 25.4 %
# against train MAPE = 5.9 %. Train and valid already clear the R2 >= 0.85 and
# MAPE < 20 % targets; only test misses. Since the model fits training data at
# 0.97, capacity is not the constraint -- the gap is generalisation across
# physical configurations, so the arms below all move regularisation, and none
# moves capacity upward.
#
# Runs on the CORRECTED 50-simulation corpus (MLP_DROP_DEFECTIVE=1), because
# tuning against ten simulations whose cladding core does not burn would be
# tuning against a known artefact. Arm A0 is the 50-run baseline, which does not
# otherwise exist, so every comparison here is like-for-like.
#
# SELECTION PROTOCOL -- this matters more than the arms.
# The winning arm is chosen on VALIDATION only. Test is read once, afterwards,
# for the chosen arm. Choosing the arm with the best test score would be fitting
# the test set across seven attempts, which is the same class of error as the
# identifier-level leakage this project already withdrew, and it would make the
# reported test figure meaningless.
#
#   .\run_regsweep.ps1                # all arms, sequential
#   .\run_regsweep.ps1 -WhatIf        # print the plan
#
# Resumable; skips any arm whose best_model.pt exists. Never wrap in Start-Job.

param([switch]$WhatIf)

$ErrorActionPreference = "Stop"
$proj = "D:\VS_projects\bs8414_MLP_surrogate"
$py   = Join-Path $proj "venv\Scripts\python.exe"
$logRoot = Join-Path $proj "outputs\regsweep"

if (-not (Test-Path $py)) { throw "venv python not found at $py" }
$trainSrc = Get-Content (Join-Path $proj "train.py") -Raw
foreach ($k in @("MLP_DROPOUT", "MLP_WD", "MLP_HIDDEN")) {
    if ($trainSrc -notmatch [regex]::Escape($k)) {
        throw "train.py lacks the $k override; the sweep would train seven identical models."
    }
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONPATH = $proj
$env:MLP_SPLIT = "70_15_15"
$env:MLP_DROP_DEFECTIVE = "1"

# name, dropout, weight decay, hidden, note
$arms = @(
    @{ n = "reg_A0_baseline";      d = "0.15"; w = "2e-4"; h = "96"; note = "defaults, 50-run corpus" },
    @{ n = "reg_A1_drop30";        d = "0.30"; w = "2e-4"; h = "96"; note = "dropout 0.30" },
    @{ n = "reg_A2_drop45";        d = "0.45"; w = "2e-4"; h = "96"; note = "dropout 0.45" },
    @{ n = "reg_A3_wd1e3";         d = "0.15"; w = "1e-3"; h = "96"; note = "weight decay 1e-3" },
    @{ n = "reg_A4_hidden64";      d = "0.15"; w = "2e-4"; h = "64"; note = "hidden 64" },
    @{ n = "reg_A5_drop30_wd1e3";  d = "0.30"; w = "1e-3"; h = "96"; note = "dropout 0.30 + wd 1e-3" },
    @{ n = "reg_A6_h64_drop30";    d = "0.30"; w = "2e-4"; h = "64"; note = "hidden 64 + dropout 0.30" }
)

Write-Output "regsweep start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "corpus: 50 simulations (10 excluded, non-combusting cladding core)"
Write-Output "targets: R2 >= 0.85 and MAPE < 20 % on train, valid AND test"
Write-Output "selection: validation only; test read once for the chosen arm`n"

foreach ($a in $arms) {
    $name = "models_mlp_" + $a.n
    $outDir = Join-Path $logRoot $a.n

    if (Test-Path (Join-Path $proj "$name\best_model.pt")) {
        Write-Host "  [skip] $($a.n) -- already trained"
        continue
    }
    if ($WhatIf) {
        Write-Host "  [plan] $($a.n)  dropout=$($a.d) wd=$($a.w) hidden=$($a.h)  ($($a.note))"
        continue
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $env:MLP_DROPOUT = $a.d
    $env:MLP_WD      = $a.w
    $env:MLP_HIDDEN  = $a.h
    Write-Host "  [run ] $($a.n)  $($a.note)  start $(Get-Date -Format 'HH:mm:ss')"
    try {
        & $py -u (Join-Path $proj "train.py") --model-dir $name `
            1> (Join-Path $outDir "train.log") 2> (Join-Path $outDir "train.err.log")
        if ($LASTEXITCODE -ne 0) { throw "train.py failed for $name (exit $LASTEXITCODE)" }

        & $py -u (Join-Path $proj "evaluate.py") --model-dir $name `
            1> (Join-Path $outDir "evaluate.log") 2> (Join-Path $outDir "evaluate.err.log")
        if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $name (exit $LASTEXITCODE)" }

        Copy-Item (Join-Path $proj "$name\outputs\overall_metrics.csv") `
                  (Join-Path $outDir "overall_metrics.csv") -Force
    }
    finally {
        Remove-Item Env:MLP_DROPOUT, Env:MLP_WD, Env:MLP_HIDDEN -ErrorAction SilentlyContinue
    }
    Write-Host "  [done] $($a.n)  $(Get-Date -Format 'HH:mm:ss')"
}

Write-Output "`nregsweep end   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "aggregate with: python collect_regsweep.py"
