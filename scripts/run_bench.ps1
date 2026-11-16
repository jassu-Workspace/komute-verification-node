# Run a benchmark in the foreground with live, per-image progress. Output is also saved to a log.
#   .\scripts\run_bench.ps1 face     # AgeDB-30, 12,000 faces
#   .\scripts\run_bench.ps1 plate    # OpenALPR US, 222 photos
#   .\scripts\run_bench.ps1 rejects  # re-run the 776 AgeDB-30 false rejects
#   .\scripts\run_bench.ps1 rejects -Threads 10   # same scores, more CPU
param([ValidateSet('face', 'plate', 'rejects')][string]$Mode = 'face', [ValidateRange(1, 64)][int]$Threads = 2)
$ErrorActionPreference = 'Stop'
$K = Join-Path $env:LOCALAPPDATA 'komute'
$Py = Join-Path $K 'venv\Scripts\python.exe'
foreach ($line in Get-Content -LiteralPath (Join-Path $K 'dev.env')) {
    if ($line -match '^\s*([A-Z0-9_]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$env:PYTHONUTF8 = '1'; $env:KOMUTE_ENV = 'dev'; $env:KOMUTE_STORAGE_ROOT = Join-Path $K 'storage'
$Bench = Join-Path $K 'bench'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($Mode -eq 'face') {
    $data = Join-Path $Bench 'agedb\agedb_30.bin'; $out = 'docs\calibration\face-agedb30.md'
} else {
    $data = Join-Path $Bench 'openalpr\repo\endtoend\us'; $out = 'docs\calibration\plate-openalpr-us.md'
}
$log = Join-Path $Bench "$Mode`_run.log"
Write-Host "Running the $Mode benchmark. Log: $log" -ForegroundColor Cyan
$ErrorActionPreference = 'Continue'  # Windows PowerShell treats any stderr line from python as fatal under Stop
if ($Mode -eq 'rejects') {
    $out = "docs\calibration\face-false-rejects-0.40-at-$env:KOMUTE_FACE_MATCH_THRESHOLD.md"
    if (-not $env:KOMUTE_FACE_MATCH_THRESHOLD) { $out = 'docs\calibration\face-false-rejects-0.40-rerun.md' }
    & $Py -u scripts\bench_face_folder.py --folder (Join-Path $Bench 'agedb\false_rejects_0.40') --out $out --threads $Threads 2>&1 | Tee-Object -FilePath $log
} else {
    & $Py -u scripts\bench_public.py $Mode --data $data --out $out 2>&1 | Tee-Object -FilePath $log
}
Write-Host "Finished: exit code $LASTEXITCODE. Report: $out" -ForegroundColor Cyan
