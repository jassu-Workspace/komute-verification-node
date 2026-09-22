<#
.SYNOPSIS
    Komüte Driver Verification Service v2 - Application Startup Script
.DESCRIPTION
    Launches the FastAPI backend microservice with RapidOCR ONNX licence OCR, YuNet face isolation,
    Vehicle ALPR engine, structured uploads storage, and the interactive web dashboard.
.EXAMPLE
    .\s.ps1
.EXAMPLE
    .\s.ps1 -Port 8080 -HostAddress 0.0.0.0
.EXAMPLE
    .\s.ps1 -RunTests
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$HostAddress = "127.0.0.1",

    [Parameter(Mandatory = $false)]
    [int]$Port = 8080,

    [Parameter(Mandatory = $false)]
    [switch]$NoReload,

    [Parameter(Mandatory = $false)]
    [switch]$RunTests,

    [Parameter(Mandatory = $false)]
    [switch]$RunDLTests,

    [Parameter(Mandatory = $false)]
    [switch]$RunPlateTests,

    [Parameter(Mandatory = $false)]
    [switch]$InstallDeps
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         🚀 KOMUTE DRIVER VERIFICATION MICROSERVICE v2               " -ForegroundColor Yellow
Write-Host "      High-Privacy Cloud Biometrics & Multi-Stage Auto-Onboarding      " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Find suitable Python executable (portable - no hardcoded user paths).
function Get-PythonPath {
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot "venv\Scripts\python.exe")
    )
    foreach ($cand in $candidates) {
        if (Test-Path -LiteralPath $cand) {
            return (Resolve-Path -LiteralPath $cand).Path
        }
    }
    # Prefer the Windows py launcher (Python 3.11 first), then PATH lookup.
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $v311 = & $pyLauncher.Source -3.11 -c "import sys; print(sys.executable)" 2>$null
        if (($LASTEXITCODE -eq 0) -and (-not [string]::IsNullOrWhiteSpace($v311))) {
            return $v311.Trim()
        }
    }
    foreach ($name in @("python3.11", "python3", "python")) {
        $pyCmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($pyCmd) {
            return $pyCmd.Source
        }
    }
    return "python"
}

$pyExe = Get-PythonPath
try {
    $pyVer = & $pyExe --version 2>&1
} catch {
    Write-Host " [X] No Python interpreter found. Install Python 3.11+ and/or run ./install.ps1 first." -ForegroundColor Red
    exit 1
}
Write-Host " [OK] Python runtime: $pyVer ($pyExe)" -ForegroundColor Gray

# 2. Ensure all required storage directories exist
$uploadsPath = Join-Path -Path $PSScriptRoot -ChildPath "backend/uploads"
$requiredDirs = @(
    $uploadsPath,
    (Join-Path -Path $PSScriptRoot -ChildPath "backend/storage/uploads"),
    (Join-Path -Path $PSScriptRoot -ChildPath "storage")
)
foreach ($dir in $requiredDirs) {
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host " [OK] Created directory: $dir" -ForegroundColor Gray
    } else {
        Write-Host " [OK] Directory verified: $dir" -ForegroundColor Gray
    }
}

# 3. Optional Dependency Installation (prefers uv, falls back to pip)
if ($InstallDeps) {
    Write-Host ""
    Write-Host " [*] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        & $uvCmd.Source pip install --python $pyExe -r requirements.txt
    } else {
        & $pyExe -m pip install -r requirements.txt
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host " [X] Failed to install dependencies." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host " [OK] Dependencies installed successfully." -ForegroundColor Green
}

# 4. Optional Pre-flight Test Suite Execution
if ($RunTests) {
    Write-Host ""
    Write-Host " [*] Running Pytest Verification Suite..." -ForegroundColor Yellow
    Set-Location -LiteralPath (Join-Path $PSScriptRoot "backend")
    & $pyExe -m pytest -v
    $testExit = $LASTEXITCODE
    Set-Location -LiteralPath $PSScriptRoot
    if ($testExit -ne 0) {
        Write-Host " [X] Test execution failed." -ForegroundColor Red
        exit $testExit
    }
    Write-Host " [OK] All test suites passed successfully!" -ForegroundColor Green
}

if ($RunDLTests) {
    Write-Host ""
    Write-Host " [*] Running 102 Driving License Brutal Verification Suite (RapidOCR ONNX)..." -ForegroundColor Yellow
    Set-Location -LiteralPath (Join-Path $PSScriptRoot "backend")
    & $pyExe run_brutal_ocr_tests.py
    $testExit = $LASTEXITCODE
    Set-Location -LiteralPath $PSScriptRoot
    if ($testExit -ne 0) {
        Write-Host " [X] Driving License test execution failed." -ForegroundColor Red
        exit $testExit
    }
    exit 0
}

if ($RunPlateTests) {
    Write-Host ""
    Write-Host " [*] Running 102 Vehicle License Plate Verification Suite (RapidOCR ONNX)..." -ForegroundColor Yellow
    Set-Location -LiteralPath (Join-Path $PSScriptRoot "backend")
    & $pyExe tests/plate_verification/run_brutal_plate_tests.py
    $testExit = $LASTEXITCODE
    Set-Location -LiteralPath $PSScriptRoot
    if ($testExit -ne 0) {
        Write-Host " [X] Vehicle Plate test execution failed." -ForegroundColor Red
        exit $testExit
    }
    exit 0
}

# Validate port range and required modules before starting the server.
if (($Port -lt 1) -or ($Port -gt 65535)) {
    Write-Host " [X] Port must be between 1 and 65535 (got $Port)." -ForegroundColor Red
    exit 1
}

& $pyExe -c "import fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host " [X] Required packages (fastapi/uvicorn) are not importable with: $pyExe" -ForegroundColor Red
    Write-Host "     Run ./install.ps1 first, then retry." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    Write-Host " [!] .env not found - server will start with defaults; live VLM keys will be missing." -ForegroundColor Yellow
}

# 5. Display Endpoints
Write-Host ""
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  Available Service Endpoints:" -ForegroundColor White
Write-Host "  * Interactive Web UI:     http://${HostAddress}:${Port}/dashboard" -ForegroundColor Cyan
Write-Host "  * Swagger API Docs:       http://${HostAddress}:${Port}/docs" -ForegroundColor Green
Write-Host "  * Health Check:           http://${HostAddress}:${Port}/health" -ForegroundColor Magenta
Write-Host "  * Uploads Storage Root:   $uploadsPath" -ForegroundColor DarkCyan
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host " [!] Starting FastAPI server on http://${HostAddress}:${Port} (Press Ctrl+C to stop)..." -ForegroundColor Yellow
Write-Host ""

# 6. Execute Application Server with chosen Python environment
Set-Location -Path ".\backend"
if ($NoReload) {
    & $pyExe -m uvicorn app.main:app --host $HostAddress --port $Port
} else {
    & $pyExe -m uvicorn app.main:app --host $HostAddress --port $Port --reload
}
