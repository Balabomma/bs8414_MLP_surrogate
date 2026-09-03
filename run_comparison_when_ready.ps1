# Wait for both run_split.ps1 chains to finish, then run the cross-split comparison.
#
# Success sentinel is the mirrored metrics tree, which run_split.ps1 only writes
# after train -> evaluate -> validate_physics -> killer_excluded have all exited 0.
# Aborts (rather than waiting out the deadline) if either training crashed.
$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot
$py = Join-Path $proj "venv\Scripts\python.exe"
$splits = @("70_15_15", "80_10_10")
$deadline = (Get-Date).AddMinutes(180)

function Ready($s) {
    $m = Join-Path $proj "outputs\split_$s\metrics"
    return (Test-Path (Join-Path $m "overall_metrics.csv")) -and
           (Test-Path (Join-Path $m "killer_excluded_metrics.txt")) -and
           (Test-Path (Join-Path $m "summary.txt"))
}

function Crashed($s) {
    $e = Join-Path $proj "outputs\split_$s\train.err.log"
    if (-not (Test-Path $e)) { return $false }
    return (Select-String -Path $e -Pattern "Traceback|CUDA out of memory" -Quiet) -eq $true
}

while ($true) {
    if ((Get-Date) -gt $deadline) { throw "timed out waiting for the split chains" }
    foreach ($s in $splits) {
        if (Crashed $s) { throw "training crashed for split $s - see outputs/split_$s/train.err.log" }
    }
    if ((Ready $splits[0]) -and (Ready $splits[1])) { break }
    Start-Sleep -Seconds 30
}

Write-Output "both chains complete $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - running comparison"
Start-Sleep -Seconds 5          # let the metrics copy settle

$cmpDir = Join-Path $proj "outputs\comparison"
New-Item -ItemType Directory -Force -Path $cmpDir | Out-Null

& $py -u (Join-Path $proj "compare_splits.py") `
    1> (Join-Path $cmpDir "compare_splits.log") 2> (Join-Path $cmpDir "compare_splits.err.log")
if ($LASTEXITCODE -ne 0) { throw "compare_splits.py failed (exit $LASTEXITCODE)" }

# Give each split its own copy of the shared comparison, so every ratio's output
# folder is self-contained.
foreach ($s in $splits) {
    Copy-Item (Join-Path $cmpDir "split_comparison.md") `
        (Join-Path $proj "outputs\split_$s\split_comparison.md") -Force
}

Write-Output "comparison complete $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Get-Content (Join-Path $cmpDir "compare_splits.log") -Tail 40
