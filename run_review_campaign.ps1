# Review-response training campaign -- the four experiments Section 8.4 lists.
#
# Each phase changes exactly ONE thing relative to the current grouped baseline,
# so every result is attributable. Phases run SEQUENTIALLY on the single GPU.
#
#   .\run_review_campaign.ps1 -Phase scaler      # C5  ~2 h   (1 run)
#   .\run_review_campaign.ps1 -Phase seeds       # C4  ~4 h   (2 runs)
#   .\run_review_campaign.ps1 -Phase ablation    # M2  ~1 h   (1 run)
#   .\run_review_campaign.ps1 -Phase loco        # C3  ~25 h  (20 runs)
#   .\run_review_campaign.ps1 -Phase all         # everything, cheapest first
#   .\run_review_campaign.ps1 -Phase loco -WhatIf   # print the plan, run nothing
#
# NEVER wrap this in Start-Job. Job sessions are children of the launching
# PowerShell process and are killed with it -- that is what destroyed replicate 2
# on 2026-08-11 (70/15/15 at candidate 8/12, ~60 min lost, nothing recoverable
# because train.py only checkpoints after all 12 candidates complete). Launch it
# as its own top-level process and leave the window open.
#
# Resumable: any run whose model directory already holds best_model.pt is
# skipped, never overwritten. Kill the window at any point and re-run the same
# command to continue.

param(
    [ValidateSet("scaler", "seeds", "ablation", "loco", "all")]
    [string]$Phase = "all",

    [string]$CampaignDir = "C:\Users\saipa\AppData\Local\Temp\claude\D--Code-space\ab78f16c-a84c-4cdd-9000-65f98ab7b088\scratchpad\campaign",

    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$proj = "D:\VS_projects\bs8414_MLP_surrogate"
$py   = Join-Path $proj "venv\Scripts\python.exe"
$logRoot = Join-Path $proj "outputs\campaign"

if (-not (Test-Path $py))   { throw "venv python not found at $py" }
if (-not (Test-Path $CampaignDir)) { throw "campaign modules not found at $CampaignDir" }

# --- preflight: the train.py hooks must be present, or the scaler/seed/ablation
# --- phases would silently reproduce the existing replicates instead of the
# --- experiment. Fail loudly rather than burn hours on a null result.
$trainSrc = Get-Content (Join-Path $proj "train.py") -Raw
$needHooks = @{
    "campaign_hooks.filter_mesh"  = "mesh filter (ablation phase)"
    "campaign_hooks.refit_scaler" = "scaler refit (scaler phase)"
    "MLP_SEED_BASE"               = "seed offset (seeds phase)"
}
$missing = @()
foreach ($k in $needHooks.Keys) {
    if ($trainSrc -notmatch [regex]::Escape($k)) { $missing += "$k  -> $($needHooks[$k])" }
}
if ($missing.Count -gt 0 -and $Phase -ne "loco") {
    Write-Output "train.py is not patched. Missing:"
    $missing | ForEach-Object { Write-Output "    $_" }
    throw "Apply PATCH_train_py.md first, or run -Phase loco (which needs no patch)."
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONPATH = "$proj;$CampaignDir"
$env:MLP_SPLIT  = "70_15_15"

# One run = one model directory. Returns $true if it actually trained.
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

    # Set, run, then clear -- so one phase can never leak configuration into the next.
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

function Phase-Scaler {
    Write-Output "`n=== C5  train-only output scaler (1 run) ==="
    Write-Output "    Compare against models_mlp_70_15_15_grouped. Any difference IS the"
    Write-Output "    leakage the pre-split scaler was contributing."
    Invoke-Run -Name "models_mlp_70_15_15_grouped_trainscaler" `
               -EnvVars @{ MLP_SCALER_SCOPE = "train" } `
               -Note "scaler fitted on training sims only"
}

function Phase-Seeds {
    Write-Output "`n=== C4  seed-varied replicates (2 runs) ==="
    Write-Output "    The existing r1-r3 share candidate seeds, so their spread excludes"
    Write-Output "    initialisation variance. These two vary it."
    foreach ($s in @(1337, 2024)) {
        Invoke-Run -Name "models_mlp_70_15_15_grouped_seed$s" `
                   -EnvVars @{ MLP_SEED_BASE = "$s" } `
                   -Note "seed base $s"
    }
}

function Phase-Ablation {
    Write-Output "`n=== M2  single-resolution ablation (1 run) ==="
    Write-Output "    20 simulations, one mesh, no siblings -- separates the irreducible"
    Write-Output "    mesh-sibling floor (72.1 degC) from learnable error."
    Invoke-Run -Name "models_mlp_singleres_M009" `
               -EnvVars @{ MLP_ABLATION_MESH = "M009" } `
               -Note "0.09 m only, grouped 14/3/3"
}

function Phase-Loco {
    Write-Output "`n=== C3  leave-one-configuration-out (20 runs, ~25 h) ==="
    Write-Output "    The headline experiment. Gives a distribution over configurations"
    Write-Output "    instead of one draw from three."
    for ($f = 0; $f -lt 20; $f++) {
        Invoke-Run -Name ("models_mlp_loco_f{0:d2}" -f $f) `
                   -EnvVars @{ MLP_CAMPAIGN = "loco"; MLP_LOCO_FOLD = "$f" } `
                   -Note "fold $f"
    }
}

Write-Output "campaign start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  phase=$Phase"
switch ($Phase) {
    "scaler"   { Phase-Scaler }
    "seeds"    { Phase-Seeds }
    "ablation" { Phase-Ablation }
    "loco"     { Phase-Loco }
    "all"      { Phase-Scaler; Phase-Ablation; Phase-Seeds; Phase-Loco }
}
Write-Output "`ncampaign end   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "logs and metrics -> $logRoot"
Write-Output "`nNext: aggregate with collect_campaign.py (LOCO distribution, scaler delta,"
Write-Output "seed-varied band, ablation floor) and fold the numbers into Sections 6.1, 7.1 and 8.4."
