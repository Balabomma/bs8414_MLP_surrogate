# Re-run of the review campaign on the corrected 50-simulation corpus.
#
# WHY: every M009 deck of the three combustible-core systems omits the &REAC
# that consumes the core fuel, so the cladding core pyrolyses but never burns.
# Ten of the sixty simulations are affected. They are excluded via
# MLP_DROP_DEFECTIVE=1 (see defective_runs.py). No FDS output is deleted.
#
# Every model directory carries a _c50 suffix, so nothing here overwrites the
# 60-run results. Those stay on disk for the before/after comparison the paper
# now needs.
#
# Two design changes forced by the smaller corpus:
#
#   baseline    a new grouped baseline is required. The 60-run baseline is not
#               a valid comparator for anything trained on 50.
#   ablation    moves from M009 to M010. After the drop M009 holds 10 runs and
#               is the stratum under suspicion; M010 holds 20, which is the
#               design size the original single-resolution ablation had.
#
#   .\run_campaign_c50.ps1 -Phase all          # everything, cheapest first
#   .\run_campaign_c50.ps1 -Phase loco         # the 20 folds only
#   .\run_campaign_c50.ps1 -Phase all -WhatIf  # print the plan, run nothing
#
# NEVER wrap this in Start-Job. Job sessions die with the launching process;
# that is what destroyed replicate 2 on 2026-08-11. Launch it top-level and
# leave the window open.
#
# Resumable: any run whose model directory already holds best_model.pt is
# skipped, never overwritten.

param(
    [ValidateSet("baseline", "scaler", "seeds", "ablation", "loco", "all")]
    [string]$Phase = "all",

    [string]$CampaignDir = "C:\Users\saipa\AppData\Local\Temp\claude\D--Code-space\ab78f16c-a84c-4cdd-9000-65f98ab7b088\scratchpad\campaign",

    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$proj = "D:\VS_projects\bs8414_MLP_surrogate"
$py   = Join-Path $proj "venv\Scripts\python.exe"
$logRoot = Join-Path $proj "outputs\campaign_c50"

if (-not (Test-Path $py)) { throw "venv python not found at $py" }

# Preflight. The drop is applied inside campaign_hooks.filter_mesh, which
# train.py and evaluate.py must both call, or the campaign would silently train
# on all 60 again and every number would be wrong in a way nothing flags.
$trainSrc = Get-Content (Join-Path $proj "train.py") -Raw
$evalSrc  = Get-Content (Join-Path $proj "evaluate.py") -Raw
foreach ($pair in @(@("train.py", $trainSrc), @("evaluate.py", $evalSrc))) {
    if ($pair[1] -notmatch [regex]::Escape("campaign_hooks.filter_mesh")) {
        throw "$($pair[0]) does not call campaign_hooks.filter_mesh; the corpus drop would not apply."
    }
}
if (-not (Test-Path (Join-Path $proj "defective_runs.py"))) {
    throw "defective_runs.py missing."
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONPATH = "$proj;$CampaignDir"
$env:MLP_SPLIT  = "70_15_15"
$env:MLP_DROP_DEFECTIVE = "1"

function Invoke-Run {
    param([string]$Name, [hashtable]$EnvVars, [string]$Note)

    $modelDir = Join-Path $proj $Name
    $outDir   = Join-Path $logRoot $Name

    if (Test-Path (Join-Path $modelDir "best_model.pt")) {
        Write-Host "  [skip] $Name -- already trained"
        return
    }
    if ($WhatIf) {
        Write-Host "  [plan] $Name  ($Note)"
        $EnvVars.GetEnumerator() | ForEach-Object { Write-Host "           $($_.Key)=$($_.Value)" }
        return
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "  [run ] $Name  ($Note)  start $(Get-Date -Format 'HH:mm:ss')"

    foreach ($kv in $EnvVars.GetEnumerator()) {
        Set-Item -Path "Env:$($kv.Key)" -Value $kv.Value
    }
    try {
        & $py -u (Join-Path $proj "train.py") --model-dir $Name `
            1> (Join-Path $outDir "train.log") 2> (Join-Path $outDir "train.err.log")
        if ($LASTEXITCODE -ne 0) { throw "train.py failed for $Name (exit $LASTEXITCODE)" }

        & $py -u (Join-Path $proj "evaluate.py") --model-dir $Name `
            1> (Join-Path $outDir "evaluate.log") 2> (Join-Path $outDir "evaluate.err.log")
        if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $Name (exit $LASTEXITCODE)" }

        $src = Join-Path $modelDir "outputs"
        $dst = Join-Path $outDir "metrics"
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
    }
    finally {
        foreach ($kv in $EnvVars.GetEnumerator()) {
            Remove-Item -Path "Env:$($kv.Key)" -ErrorAction SilentlyContinue
        }
    }
    Write-Host "  [done] $Name  $(Get-Date -Format 'HH:mm:ss')"
}

function Phase-Baseline {
    Write-Output "`n=== baseline on 50 runs (1 run) ==="
    Write-Output "    Replaces models_mlp_70_15_15_grouped as the comparator for"
    Write-Output "    everything below. Split is 33/9/8, still configuration-grouped."
    Invoke-Run -Name "models_mlp_70_15_15_grouped_c50" -EnvVars @{} `
               -Note "grouped baseline, 50-run corpus"
}

function Phase-Scaler {
    Write-Output "`n=== C5  train-only output scaler (1 run) ==="
    Invoke-Run -Name "models_mlp_70_15_15_grouped_c50_trainscaler" `
               -EnvVars @{ MLP_SCALER_SCOPE = "train" } `
               -Note "scaler fitted on training sims only"
}

function Phase-Seeds {
    Write-Output "`n=== C4  seed-varied replicates (2 runs) ==="
    foreach ($s in @(1337, 2024)) {
        Invoke-Run -Name "models_mlp_70_15_15_grouped_c50_seed$s" `
                   -EnvVars @{ MLP_SEED_BASE = "$s" } `
                   -Note "seed base $s"
    }
}

function Phase-Ablation {
    Write-Output "`n=== M2  single-resolution ablation (1 run) ==="
    Write-Output "    M010, 20 simulations, no mesh siblings. M009 is not usable"
    Write-Output "    for this after the drop: 10 runs, and the suspect stratum."
    Invoke-Run -Name "models_mlp_singleres_M010_c50" `
               -EnvVars @{ MLP_ABLATION_MESH = "M010" } `
               -Note "0.10 m only, grouped"
}

function Phase-Loco {
    Write-Output "`n=== C3  leave-one-configuration-out (20 runs) ==="
    Write-Output "    Still 20 configurations; the folds now hold out 2 or 3"
    Write-Output "    simulations depending on whether M009 survived for that config."
    for ($f = 0; $f -lt 20; $f++) {
        Invoke-Run -Name ("models_mlp_loco_c50_f{0:d2}" -f $f) `
                   -EnvVars @{ MLP_CAMPAIGN = "loco"; MLP_LOCO_FOLD = "$f" } `
                   -Note "fold $f"
    }
}

Write-Output "campaign(c50) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  phase=$Phase"
Write-Output "corpus: 50 simulations (10 excluded, non-combusting cladding core)"
switch ($Phase) {
    "baseline" { Phase-Baseline }
    "scaler"   { Phase-Scaler }
    "seeds"    { Phase-Seeds }
    "ablation" { Phase-Ablation }
    "loco"     { Phase-Loco }
    "all"      { Phase-Baseline; Phase-Scaler; Phase-Ablation; Phase-Seeds; Phase-Loco }
}
Write-Output "`ncampaign(c50) end   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "logs and metrics -> $logRoot"
