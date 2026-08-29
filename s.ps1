<#
.SYNOPSIS
    Komüte Driver Verification Service v2 - Application Startup Script
.DESCRIPTION
    Launches the FastAPI backend microservice with docTR PARSeq OCR, YuNet Deep Face Biometrics,
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

Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         🚀 KOMUTE DRIVER VERIFICATION MICROSERVICE v2               " -ForegroundColor Yellow
Write-Host "      High-Privacy Cloud Biometrics & Multi-Stage Auto-Onboarding      " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Find suitable Python executable with required libraries
function Get-PythonPath {
    $candidatePaths = @(
        ".\.venv\Scripts\python.exe",
        ".\venv\Scripts\python.exe",
        "C:\Users\jaswa\AppData\Local\Programs\Python\Python311\python.exe"
    )
    foreach ($cand in $candidatePaths) {
        if (Test-Path -Path $cand) {
            return (Resolve-Path $cand).Path
        }
    }
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return $pyCmd.Source
    }
    return "python"
}

$pyExe = Get-PythonPath
$pyVer = & $pyExe --version 2>&1
Write-Host " [OK] Python runtime: $pyVer ($pyExe)" -ForegroundColor Gray

# 2. Check and ensure uploads directory exists
$uploadsPath = Join-Path -Path $PSScriptRoot -ChildPath "backend/uploads"
if (-not (Test-Path -Path $uploadsPath)) {
    New-Item -ItemType Directory -Path $uploadsPath -Force | Out-Null
    Write-Host " [OK] Created storage directory: $uploadsPath" -ForegroundColor Gray
} else {
    Write-Host " [OK] Storage directory verified: $uploadsPath" -ForegroundColor Gray
}

# 3. Optional Dependency Installation
if ($InstallDeps) {
    Write-Host ""
    Write-Host " [*] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & $pyExe -m pip install -r requirements.txt
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
    Set-Location -Path ".\backend"
    & $pyExe -m pytest -v
    $testExit = $LASTEXITCODE
    Set-Location -Path $PSScriptRoot
    if ($testExit -ne 0) {
        Write-Host " [X] Test execution failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host " [OK] All test suites passed successfully!" -ForegroundColor Green
}

if ($RunDLTests) {
    Write-Host ""
    Write-Host " [*] Running 102 Driving License Brutal Verification Suite (RapidOCR ONNX)..." -ForegroundColor Yellow
    Set-Location -Path ".\backend"
    & $pyExe run_brutal_ocr_tests.py
    $testExit = $LASTEXITCODE
    Set-Location -Path $PSScriptRoot
    if ($testExit -ne 0) {
        Write-Host " [X] Driving License test execution failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    exit 0
}

if ($RunPlateTests) {
    Write-Host ""
    Write-Host " [*] Running 102 Vehicle License Plate Verification Suite (RapidOCR ONNX)..." -ForegroundColor Yellow
    Set-Location -Path ".\backend"
    & $pyExe tests/plate_verification/run_brutal_plate_tests.py
    $testExit = $LASTEXITCODE
    Set-Location -Path $PSScriptRoot
    if ($testExit -ne 0) {
        Write-Host " [X] Vehicle Plate test execution failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    exit 0
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
