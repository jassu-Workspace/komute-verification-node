<#
.SYNOPSIS
  Checks the environment, then starts the Komute-Local backend and opens the test page.

.DESCRIPTION
  1. Finds uv and Python 3.12 (creates the venv with uv if needed).
  2. Checks every runtime dependency; if any is missing, installs requirements.txt with uv.
  3. Checks that every locked model file is present and matches backend\models\models.lock.
  4. Creates dev keys on first run, starts the server on 127.0.0.1, opens the page in the browser.

  Stop the server with Ctrl+C.

.EXAMPLE
  .\run.ps1
  .\run.ps1 -Port 8010 -NoBrowser
  .\run.ps1 -CheckOnly
  .\run.ps1 -WithApiKey      # require the X-API-Key header (key: %LOCALAPPDATA%\komute\dev.env)
  .\run.ps1 -ApiKey komute-local-TEST-key-not-for-production   # require this key (Postman "auth on" environment)
  powershell -ExecutionPolicy Bypass -File .\run.ps1    # if script execution is blocked
#>
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$WithApiKey,
    [string]$ApiKey = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$Home_ = Join-Path $env:LOCALAPPDATA 'komute'
$VenvDir = Join-Path $Home_ 'venv'
$Py = Join-Path $VenvDir 'Scripts\python.exe'
$DevEnv = Join-Path $Home_ 'dev.env'
$Requirements = Join-Path $PSScriptRoot 'requirements.txt'
$ModelsDir = Join-Path $PSScriptRoot 'backend\models'
$LockFile = Join-Path $ModelsDir 'models.lock'

# Modules the backend imports at runtime (import name -> shown in messages).
$RuntimeImports = @('fastapi', 'uvicorn', 'pydantic', 'pydantic_settings', 'multipart', 'psutil', 'cv2',
                    'numpy', 'onnxruntime', 'zxingcpp', 'rapidfuzz', 'scipy', 'cryptography', 'tzdata')

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    ok  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    !!  $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "`nERROR: $text" -ForegroundColor Red; exit 1 }

function Invoke-Python {
    # Runs the venv python quietly; returns exit code. Output goes to $script:LastOutput.
    param([string[]]$Arguments)
    $script:LastOutput = & $Py @Arguments 2>&1 | Out-String
    return $LASTEXITCODE
}

# ---------------------------------------------------------------- 1. uv and Python 3.12
Step 'Checking uv and Python 3.12'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail ("uv is not installed. Install it, then run this script again:`n" +
          "    powershell -ExecutionPolicy ByPass -c ""irm https://astral.sh/uv/install.ps1 | iex""")
}
Ok ("uv " + ((& uv --version) -replace '^uv ', ''))

$venvOk = $false
if (Test-Path -LiteralPath $Py) {
    $v = (& $Py -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
    $venvOk = ($v -eq '3.12')
}
if (-not $venvOk) {
    Warn "No Python 3.12 virtual environment at $VenvDir. Creating it with uv."
    & uv python install 3.12
    if ($LASTEXITCODE -ne 0) { Fail 'uv could not install Python 3.12.' }
    & uv venv --python 3.12 --allow-existing $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Py)) { Fail 'uv could not create the virtual environment.' }
}
Ok ("Python " + (& $Py -c "import sys; print(sys.version.split()[0])") + "  ($VenvDir)")

# ---------------------------------------------------------------- 2. dependencies
Step 'Checking dependencies'
if (-not (Test-Path -LiteralPath $Requirements)) { Fail "requirements.txt not found at $Requirements" }

function Get-MissingImports {
    $code = "import importlib.util, sys; print(' '.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
    $out = & $Py -c $code @RuntimeImports 2>&1 | Out-String
    return @($out.Trim() -split '\s+' | Where-Object { $_ })
}

$missing = Get-MissingImports
if ($missing.Count -gt 0) {
    Warn ("Missing: " + ($missing -join ', ') + ". Installing requirements.txt with uv.")
    & uv pip install --python $Py -r $Requirements
    if ($LASTEXITCODE -ne 0) { Fail 'uv pip install failed. Fix the error above and run again.' }
    $missing = Get-MissingImports
    if ($missing.Count -gt 0) { Fail ("Still missing after install: " + ($missing -join ', ')) }
}
Ok ("all " + $RuntimeImports.Count + " runtime modules are installed")

# Importing cv2 before backend.core would disable the pixel-limit guard: check the guard itself.
$rc = Invoke-Python @('-c', 'import backend.core')
if ($rc -ne 0) { Fail ("The backend package does not import:`n" + $LastOutput.Trim()) }
Ok 'backend package imports (thread and pixel guards active)'

# ---------------------------------------------------------------- 3. models
Step 'Checking model files'
if (-not (Test-Path -LiteralPath $LockFile)) { Fail "models.lock not found at $LockFile" }
$lock = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json
$names = @($lock.PSObject.Properties.Name)
$absent = @($names | Where-Object { -not (Test-Path -LiteralPath (Join-Path $ModelsDir $_)) })
if ($absent.Count -gt 0) {
    Fail ("These model files are missing from $ModelsDir`:`n    " + ($absent -join "`n    ") +
          "`nRun .\setup.ps1 to download and hash-verify them, then run again.")
}
$verify = "from pathlib import Path; from backend.core.registry import verify_lock; " +
          "verify_lock(Path('backend/models'), Path('backend/models/models.lock'))"
$rc = Invoke-Python @('-c', $verify)
if ($rc -ne 0) {
    Fail ("A model file does not match backend\models\models.lock (changed or corrupted):`n" + $LastOutput.Trim())
}
Ok ("$($names.Count) locked model files present and hash-verified")

# ---------------------------------------------------------------- 4. keys and settings
Step 'Preparing settings'
if (-not (Test-Path -LiteralPath $DevEnv)) {
    Warn 'No dev keys yet. Creating them (once).'
    & $Py (Join-Path $PSScriptRoot 'scripts\make_dev_keys.py')
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $DevEnv)) { Fail 'Could not create dev keys.' }
}
foreach ($line in Get-Content -LiteralPath $DevEnv) {
    if ($line -match '^\s*([A-Z0-9_]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
if ($ApiKey) {
    if ($ApiKey.Length -lt 24) { Fail '-ApiKey must be at least 24 characters.' }
    $env:KOMUTE_API_KEY = $ApiKey
    Ok 'authentication: ON with the key given in -ApiKey (callers must send the X-API-Key header)'
} elseif ($WithApiKey) {
    Ok 'authentication: ON (callers must send the X-API-Key header; the key is in dev.env)'
} else {
    Remove-Item Env:KOMUTE_API_KEY -ErrorAction SilentlyContinue
    Warn 'authentication: OFF (testing). Use -WithApiKey to require the X-API-Key header.'
}
$env:PYTHONUTF8 = '1'
$env:KOMUTE_ENV = 'dev'
if (-not $env:KOMUTE_RESPONSE_BARCODE_FIELDS) { $env:KOMUTE_RESPONSE_BARCODE_FIELDS = 'true' }  # the test page shows all decoded fields
if (-not $env:KOMUTE_STORAGE_ROOT) { $env:KOMUTE_STORAGE_ROOT = Join-Path $Home_ 'storage' }
if ($env:KOMUTE_STORAGE_ROOT -match 'OneDrive') { Fail "KOMUTE_STORAGE_ROOT must not be inside OneDrive: $env:KOMUTE_STORAGE_ROOT" }
if ($env:KOMUTE_LIVENESS_MODEL) {
    Ok "liveness model: $env:KOMUTE_LIVENESS_MODEL"
} else {
    $env:KOMUTE_DEV_SKIP_LIVENESS = 'true'
    Warn 'No liveness model: liveness is SKIPPED (development only; responses carry LIVENESS_SKIPPED_DEV).'
}
Ok "storage: $env:KOMUTE_STORAGE_ROOT"

if ($CheckOnly) { Write-Host "`nAll checks passed." -ForegroundColor Green; exit 0 }

# ---------------------------------------------------------------- 5. start the server
Step "Starting the backend on http://127.0.0.1:$Port"
# Stop an earlier Komute server on this port. Found by command line, because the port owner is not always readable.
$old = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'backend\.app\.main' -and $_.CommandLine -match "--port $Port( |$)" })
foreach ($o in $old) {
    Warn "Stopping the earlier Komute server on port $Port (PID $($o.ProcessId))."
    cmd.exe /c "taskkill /PID $($o.ProcessId) /T /F >nul 2>&1"   # /T also ends the launcher and worker; a PID already gone is fine
}
for ($i = 0; $i -lt 60 -and (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 50 }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Fail "Port $Port is used by another program, not a Komute server. Stop it yourself or run: .\run.ps1 -Port 8010"
}
$server = Start-Process -FilePath $Py -PassThru -NoNewWindow -WorkingDirectory $PSScriptRoot -ArgumentList @(
    '-m', 'uvicorn', 'backend.app.main:create_app', '--factory',
    '--host', '127.0.0.1', '--port', "$Port", '--workers', '1')

try {
    $url = "http://127.0.0.1:$Port"
    $ready = $false
    for ($i = 0; $i -lt 120 -and -not $ready; $i++) {
        if ($server.HasExited) { Fail "The server stopped during startup (exit code $($server.ExitCode)). See the messages above." }
        try {
            $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 2
            $ready = ($r.StatusCode -eq 200)
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { Fail 'The server did not become healthy within 60 seconds.' }
    Ok 'models loaded, server healthy'

    if (-not $NoBrowser) { Start-Process $url }
    Write-Host "`nRunning at $url   (Ctrl+C to stop)" -ForegroundColor Green
    Wait-Process -Id $server.Id
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        Write-Host "`nServer stopped." -ForegroundColor Cyan
    }
}
