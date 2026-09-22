<#
.SYNOPSIS
    Komute Driver Verifier v2 - One-command installer (uv + all dependencies).
.DESCRIPTION
    Run once from the repo root:
        ./install.ps1
    What it does, in order:
      1. Installs `uv` (Astral) for the current user if missing.
      2. Creates `.venv` (prefers Python 3.11) if missing.
      3. Installs every package from requirements.txt ONE AFTER ANOTHER
         using `uv pip install` with pip fallback.
      4. Verifies critical imports and creates required folders.
.NOTES
    Idempotent - safe to re-run. Stops on first fatal error.
.EXAMPLE
    ./install.ps1
.EXAMPLE
    ./install.ps1 -Recreate
.EXAMPLE
    ./install.ps1 -PythonVersion 3.11 -NoVerify
#>

[CmdletBinding()]
param (
    [string]$PythonVersion = "3.11",
    [switch]$Recreate,
    [switch]$NoVerify,
    [string]$RequirementsFile = "requirements.txt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Always operate from the script's directory (repo root).
Set-Location -LiteralPath $PSScriptRoot

$RequirementsPath = Join-Path -Path $PSScriptRoot -ChildPath $RequirementsFile
$VenvPath         = Join-Path -Path $PSScriptRoot -ChildPath ".venv"
$VenvPython       = Join-Path -Path $VenvPath -ChildPath "Scripts\python.exe"

function Write-Step([string]$Msg) {
    Write-Host ""
    Write-Host " ==> $Msg" -ForegroundColor Cyan
}

function Write-Ok([string]$Msg) {
    Write-Host " [OK] $Msg" -ForegroundColor Green
}

function Write-Info([string]$Msg) {
    Write-Host " [...] $Msg" -ForegroundColor Gray
}

function Write-Warn([string]$Msg) {
    Write-Host " [!] $Msg" -ForegroundColor Yellow
}

function Fail([string]$Msg, [int]$Code = 1) {
    Write-Host " [X] $Msg" -ForegroundColor Red
    exit $Code
}

# ------------------------------------------------------------------
# 0. Pre-flight checks
# ------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   KOMUTE DRIVER VERIFIER v2 - ONE-COMMAND INSTALLER (uv based)        " -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    Fail "requirements.txt not found at: $RequirementsPath"
}
Write-Ok "Repo root: $PSScriptRoot"
Write-Ok "Requirements file: $RequirementsPath"

# TLS 1.2 is required for the uv download on Windows PowerShell 5.1.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

# Cache dir and repo may live on different filesystems - copy mode avoids the
# "Failed to hardlink files; falling back to full copy" warning on some laptops.
$env:UV_LINK_MODE = "copy"

# ------------------------------------------------------------------
# 1. Install uv if missing
# ------------------------------------------------------------------
Write-Step "Step 1/4 - Ensuring 'uv' is installed..."

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCmd) {
    # Also check the well-known install locations before downloading.
    $wellKnown = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $HOME ".cargo\bin\uv.exe")
    )
    foreach ($p in $wellKnown) {
        if (Test-Path -LiteralPath $p) {
            $uvCmd = Get-Command $p -ErrorAction SilentlyContinue
            break
        }
    }
}

if ($null -eq $uvCmd) {
    Write-Info "'uv' not found. Installing via official Astral installer..."
    try {
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString("https://astral.sh/uv/install.ps1"))
    } catch {
        Fail "Failed to download/install uv: $($_.Exception.Message)"
    }

    # The installer drops uv.exe into $HOME\.local\bin - add it to this session's PATH.
    $uvBinDir = Join-Path $HOME ".local\bin"
    if (($env:Path -split ";") -notcontains $uvBinDir) {
        $env:Path = "$uvBinDir;$env:Path"
    }
    $cargoBinDir = Join-Path $HOME ".cargo\bin"
    if ((Test-Path -LiteralPath $cargoBinDir) -and (($env:Path -split ";") -notcontains $cargoBinDir)) {
        $env:Path = "$cargoBinDir;$env:Path"
    }

    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $uvCmd) {
        Fail "'uv' installed but not found on PATH. Restart the terminal, then re-run ./install.ps1. Expected at: $uvBinDir\uv.exe"
    }
}

$uvExe = $uvCmd.Source
Write-Ok "uv found: $uvExe"
& $uvExe --version
if ($LASTEXITCODE -ne 0) { Fail "'uv --version' failed." $LASTEXITCODE }
Write-Ok "uv is working."

# ------------------------------------------------------------------
# 2. Ensure .venv exists (prefers Python 3.11)
# ------------------------------------------------------------------
Write-Step "Step 2/4 - Ensuring virtual environment (.venv)..."

if ($Recreate -and (Test-Path -LiteralPath $VenvPath)) {
    Write-Info "-Recreate supplied. Removing existing .venv..."
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Info "Creating .venv (uv venv --python $PythonVersion)..."

    # Try pinned Python first, fall back to whatever uv/python can find.
    & $uvExe venv ".venv" --python $PythonVersion
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path -LiteralPath $VenvPython))) {
        Write-Warn "uv venv --python $PythonVersion failed. Retrying with default Python..."
        & $uvExe venv ".venv"
    }
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path -LiteralPath $VenvPython))) {
        Write-Warn "uv venv failed. Falling back to 'python -m venv .venv'..."
        $sysPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $sysPython) { Fail "No 'python' found on PATH. Install Python $PythonVersion+ from https://www.python.org/downloads/ and re-run." }
        & $sysPython.Source -m venv ".venv"
        if (($LASTEXITCODE -ne 0) -or (-not (Test-Path -LiteralPath $VenvPython))) {
            Fail "Could not create .venv."
        }
    }
    Write-Ok ".venv created."
} else {
    Write-Ok ".venv already exists. Reusing it."
}

$pyVer = & $VenvPython --version 2>&1
Write-Ok "Venv Python: $pyVer ($VenvPython)"

# Ensure pip exists inside the venv (uv venv normally includes it).
& $VenvPython -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Info "Bootstrapping pip inside .venv..."
    & $VenvPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { Fail "ensurepip failed." $LASTEXITCODE }
}

# ------------------------------------------------------------------
# 3. Install dependencies ONE AFTER ANOTHER
# ------------------------------------------------------------------
Write-Step "Step 3/4 - Installing dependencies one after another..."

# Parse requirements.txt: strip comments, blank lines, -r / --option lines.
$rawLines = Get-Content -LiteralPath $RequirementsPath
$packages = @()
foreach ($line in $rawLines) {
    $t = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($t)) { continue }
    if ($t.StartsWith("#")) { continue }
    if ($t.StartsWith("-r") -or $t.StartsWith("--")) {
        Write-Warn "Skipping directive line (not supported by per-package loop): $t"
        continue
    }
    # Strip inline comments ("package  # comment").
    $hashIdx = $t.IndexOf("#")
    if ($hashIdx -ge 0) { $t = $t.Substring(0, $hashIdx).Trim() }
    if ([string]::IsNullOrWhiteSpace($t)) { continue }
    $packages += $t
}

if ($packages.Count -eq 0) { Fail "No installable packages found in requirements.txt." }

Write-Info "$($packages.Count) package(s) to install, in order."
$i = 0
$failed = @()
foreach ($pkg in $packages) {
    $i++
    Write-Host ""
    Write-Host " [$i/$($packages.Count)] Installing: $pkg" -ForegroundColor Yellow

    & $uvExe pip install --python $VenvPython "$pkg"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "uv failed for '$pkg'. Retrying with venv pip..."
        & $VenvPython -m pip install "$pkg"
        if ($LASTEXITCODE -ne 0) {
            Write-Host " [X] FAILED: $pkg" -ForegroundColor Red
            $failed += $pkg
            continue
        }
    }
    Write-Host " [OK] Done: $pkg" -ForegroundColor Green
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host " [X] $($failed.Count) package(s) failed:" -ForegroundColor Red
    foreach ($f in $failed) { Write-Host "     - $f" -ForegroundColor Red }
    Fail "Dependency installation incomplete. Fix the lines above and re-run ./install.ps1 (it resumes safely)."
}
Write-Ok "All $($packages.Count) dependencies installed."

# ------------------------------------------------------------------
# 4. Verify + scaffold folders
# ------------------------------------------------------------------
Write-Step "Step 4/4 - Verifying installation..."

$dirs = @(
    "backend\uploads",
    "backend\storage\uploads",
    "storage"
)
foreach ($d in $dirs) {
    $full = Join-Path -Path $PSScriptRoot -ChildPath $d
    if (-not (Test-Path -LiteralPath $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Info "Created: $full"
    }
}
Write-Ok "Storage directories verified."

$envFile = Join-Path -Path $PSScriptRoot -ChildPath ".env"
if (Test-Path -LiteralPath $envFile) {
    Write-Ok ".env exists."
} else {
    Write-Warn ".env not found. Copy it from your template before running the server."
}

if (-not $NoVerify) {
    Write-Info "Import check (fastapi, uvicorn, cv2, numpy, PIL, rapidocr, onnxruntime, rapidfuzz)..."
    $checkCode = "import fastapi, uvicorn, cv2, numpy, PIL, onnxruntime, rapidfuzz; print('imports-ok')"
    & $VenvPython -c $checkCode
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Core import check failed. The server may still run, but check the errors above."
        Write-Info "Tip: re-run ./install.ps1 or run: .\.venv\Scripts\python.exe -c `"$checkCode`" for details."
    } else {
        Write-Ok "Core imports verified."
    }
} else {
    Write-Info "Import check skipped (-NoVerify)."
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   INSTALL COMPLETE. Next steps:                                      " -ForegroundColor White
Write-Host "     1. Activate:  .\.venv\Scripts\Activate.ps1                       " -ForegroundColor Cyan
Write-Host "     2. Run:       .\s.ps1                                            " -ForegroundColor Cyan
Write-Host "     3. Docs:      http://127.0.0.1:8080/docs                         " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
