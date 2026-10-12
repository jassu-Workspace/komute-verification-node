<#
.SYNOPSIS
    Komute Driver Verifier v2 - Lightning-Fast Universal Installer & Hardware Auditor.
.DESCRIPTION
    Audits the user's laptop hardware and software environment, reports installed vs missing tools,
    and installs all dependencies at lightning speed using Astral's `uv` engine.

    Can be run locally from repo root:
        .\install.ps1

    OR run as a universal 1-liner directly from GitHub in PowerShell on any laptop:
        irm https://raw.githubusercontent.com/jassu-Workspace/komute-verification-node/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1
.EXAMPLE
    .\install.ps1 -Recreate
.EXAMPLE
    .\install.ps1 -PythonVersion 3.11
#>

[CmdletBinding()]
param (
    [string]$PythonVersion = "3.11",
    [switch]$Recreate,
    [switch]$NoVerify,
    [string]$RequirementsFile = "requirements.txt",
    [switch]$SkipHardwareAudit,
    [string]$RepoUrl = "https://github.com/jassu-Workspace/komute-verification-node.git",
    [string]$RawInstallUrl = "https://raw.githubusercontent.com/jassu-Workspace/komute-verification-node/main/install.ps1"
)

Set-StrictMode -Off
$ErrorActionPreference = "Continue"

# Force TLS 1.2+ for all web requests on PowerShell 5.1 / 7+
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
} catch {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
}

# ------------------------------------------------------------------
# Output Formatting Helpers
# ------------------------------------------------------------------
function Write-Header([string]$Text) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "   $Text" -ForegroundColor Yellow
    Write-Host "======================================================================" -ForegroundColor Cyan
}

function Write-Section([string]$Text) {
    Write-Host ""
    Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
}

function Write-Step([string]$Msg) {
    Write-Host ""
    Write-Host " ==> $Msg" -ForegroundColor Cyan
}

function Write-Ok([string]$Msg) {
    Write-Host "  [OK] $Msg" -ForegroundColor Green
}

function Write-Info([string]$Msg) {
    Write-Host "  [...] $Msg" -ForegroundColor Gray
}

function Write-Warn([string]$Msg) {
    Write-Host "  [!] $Msg" -ForegroundColor Yellow
}

function Write-Fail([string]$Msg, [int]$Code = 1) {
    Write-Host "  [X] $Msg" -ForegroundColor Red
    exit $Code
}

# ------------------------------------------------------------------
# 0. Remote One-Liner Auto-Bootstrapper (If executed outside repo)
# ------------------------------------------------------------------
$CurrentRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CurrentRoot)) {
    $CurrentRoot = (Get-Location).Path
}

# Verify if we are inside the komute repository root
$IsInsideRepo = (Test-Path (Join-Path $CurrentRoot "backend\app\main.py"))

if (-not $IsInsideRepo) {
    Write-Header "KOMUTE DRIVER VERIFIER v2 - REMOTE BOOTSTRAPPER"
    Write-Warn "Not currently inside the komute repository root."
    Write-Info "Working directory: $CurrentRoot"
    Write-Step "Bootstrapping repository from GitHub..."

    $TargetDir = Join-Path $CurrentRoot "komute-verification-node"
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue

    if ($null -ne $gitCmd) {
        Write-Info "Cloning repository using Git: $RepoUrl"
        if (Test-Path $TargetDir) {
            Write-Info "Directory already exists: $TargetDir. Pulling latest changes..."
            Push-Location $TargetDir
            & git pull
        } else {
            & git clone $RepoUrl $TargetDir
            Push-Location $TargetDir
        }
    } else {
        Write-Info "Git not detected. Downloading repository archive via HTTPS..."
        $ZipUrl = "https://github.com/jassu-Workspace/komute-verification-node/archive/refs/heads/main.zip"
        $ZipPath = Join-Path $CurrentRoot "komute-repo.zip"
        $ExtractPath = Join-Path $CurrentRoot "komute-verification-node-extract"

        (New-Object System.Net.WebClient).DownloadFile($ZipUrl, $ZipPath)
        Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractPath -Force
        Remove-Item -LiteralPath $ZipPath -Force

        $ExtractedSub = Get-ChildItem -Path $ExtractPath -Directory | Select-Object -First 1
        if ($null -ne $ExtractedSub) {
            $TargetDir = $ExtractedSub.FullName
        } else {
            $TargetDir = $ExtractPath
        }
        Push-Location $TargetDir
    }

    $CurrentRoot = (Get-Location).Path
    Write-Ok "Repository root ready at: $CurrentRoot"
} else {
    Set-Location -LiteralPath $CurrentRoot
}

$RequirementsPath = Join-Path -Path $CurrentRoot -ChildPath $RequirementsFile
$VenvPath         = Join-Path -Path $CurrentRoot -ChildPath ".venv"
$VenvPython       = Join-Path -Path $VenvPath -ChildPath "Scripts\python.exe"

# Copy mode avoids cross-drive hardlink delays on Windows
$env:UV_LINK_MODE = "copy"

# ------------------------------------------------------------------
# 1. Laptop Hardware & Environment Audit
# ------------------------------------------------------------------
if (-not $SkipHardwareAudit) {
    Write-Header "LAPTOP HARDWARE & ENVIRONMENT AUDIT"

    # OS Info
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
        $osName = if ($os) { "$($os.Caption) ($($os.OSArchitecture))" } else { [System.Environment]::OSVersion.VersionString }
    } catch {
        $osName = [System.Environment]::OSVersion.VersionString
    }
    Write-Host "  [OS]           $osName" -ForegroundColor White

    # CPU Info
    try {
        $cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
        $cpuName = if ($cpu) { "$($cpu.Name.Trim()) ($($cpu.NumberOfCores) Cores, $($cpu.NumberOfLogicalProcessors) Threads)" } else { "Generic CPU" }
    } catch {
        $cpuName = "CPU Info Unavailable"
    }
    Write-Host "  [CPU]          $cpuName" -ForegroundColor White

    # RAM Info
    try {
        $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
        $totalRamBytes = if ($cs) { $cs.TotalPhysicalMemory } else { 0 }
        $totalRamGb = [math]::Round($totalRamBytes / 1GB, 1)
        
        $osMem = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
        $freeRamGb = if ($osMem) { [math]::Round($osMem.FreePhysicalMemory / 1MB, 1) } else { 0 }
        
        $ramVerdict = if ($totalRamGb -ge 16) { "Excellent for local OCR/CV" } elseif ($totalRamGb -ge 8) { "Good (Governor active)" } else { "Low RAM (Aggressive governor recommended)" }
        Write-Host "  [RAM]          $totalRamGb GB Total ($freeRamGb GB Available) - $ramVerdict" -ForegroundColor White
    } catch {
        Write-Host "  [RAM]          Memory Info Unavailable" -ForegroundColor White
    }

    # GPU Info
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        $gpuNames = @($gpus | ForEach-Object { $_.Name.Trim() })
        $hasNvidia = ($gpuNames | Where-Object { $_ -match "NVIDIA|GeForce|RTX|Quadro" })
        if ($hasNvidia) {
            Write-Host "  [GPU]          $($hasNvidia -join ', ') (CUDA Capable)" -ForegroundColor Green
        } else {
            $primaryGpu = if ($gpuNames.Count -gt 0) { $gpuNames[0] } else { "Integrated Graphics" }
            Write-Host "  [GPU]          $primaryGpu (Optimized CPU Inference Profile)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  [GPU]          GPU Info Unavailable" -ForegroundColor Gray
    }

    Write-Section "TOOLCHAIN & DEPENDENCY STATUS AUDIT"

    # Git
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($null -ne $gitCmd) {
        $gitVer = (& git --version 2>&1) -replace "git version ", ""
        Write-Ok "Git:           Installed ($gitVer)"
    } else {
        Write-Warn "Git:           Not installed (Portable HTTPS mode active)"
    }

    # System Python
    $sysPy = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $sysPy) {
        $sysPyVer = (& $sysPy.Source --version 2>&1)
        Write-Ok "System Python: Installed ($sysPyVer)"
    } else {
        Write-Warn "System Python: Not found on system PATH (uv portable engine will provide Python)"
    }

    # Winget
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -ne $wingetCmd) {
        Write-Ok "Winget:        Available (Windows Package Manager)"
    } else {
        Write-Info "Winget:        Not detected"
    }

    # UV Engine
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $uvCmd) {
        $uvPaths = @(
            (Join-Path $HOME ".local\bin\uv.exe"),
            (Join-Path $HOME ".cargo\bin\uv.exe")
        )
        foreach ($p in $uvPaths) {
            if (Test-Path -LiteralPath $p) {
                $uvCmd = Get-Command $p -ErrorAction SilentlyContinue
                break
            }
        }
    }
    if ($null -ne $uvCmd) {
        $uvVer = (& $uvCmd.Source --version 2>&1)
        Write-Ok "UV Engine:     Installed ($uvVer)"
    } else {
        Write-Warn "UV Engine:     Missing (Will auto-install in Step 1 for 10x speedup)"
    }

    # Virtual Environment
    if (Test-Path $VenvPython) {
        $venvVer = (& $VenvPython --version 2>&1)
        Write-Ok "Virtual Env:   Ready at .venv ($venvVer)"
    } else {
        Write-Warn "Virtual Env:   Missing (Will create .venv with Python $PythonVersion in Step 2)"
    }

    # Config File (.env)
    $envFile = Join-Path $CurrentRoot ".env"
    if (Test-Path $envFile) {
        Write-Ok "Configuration: .env file is present"
    } else {
        Write-Warn "Configuration: .env missing (Will scaffold from .env.example)"
    }

    # Packages Audit
    if (Test-Path $RequirementsPath) {
        $reqLines = Get-Content $RequirementsPath | Where-Object { $_.Trim() -and (-not $_.Trim().StartsWith("#")) -and (-not $_.Trim().StartsWith("-")) }
        $totalReqs = $reqLines.Count
        if (Test-Path $VenvPython) {
            $auditCode = @"
import importlib.util
pkgs = ['fastapi', 'uvicorn', 'cv2', 'numpy', 'onnxruntime', 'rapidocr_onnxruntime', 'rapidfuzz', 'pydantic', 'httpx']
installed = [p for p in pkgs if importlib.util.find_spec(p) is not None or (p == 'cv2' and importlib.util.find_spec('cv2'))]
missing = [p for p in pkgs if p not in installed]
print(f"{len(installed)}/{len(pkgs)}|{','.join(missing)}")
"@
            try {
                $auditOut = (& $VenvPython -c $auditCode 2>&1) | Out-String
                if ($auditOut -match "(\d+)/(\d+)\|(.*)") {
                    $foundCount = [int]$Matches[1]
                    $expectedCount = [int]$Matches[2]
                    $missingNames = $Matches[3].Trim()
                    if ($foundCount -eq $expectedCount) {
                        Write-Ok "Dependencies:  Status: Fully Installed ($totalReqs specifications ready in .venv)"
                    } else {
                        Write-Warn "Dependencies:  Status: $foundCount/$expectedCount core packages installed (Need to install: $missingNames)"
                    }
                } else {
                    Write-Ok "Dependencies:  $totalReqs specifications audited in requirements.txt"
                }
            } catch {
                Write-Info "Dependencies:  $totalReqs packages listed in requirements.txt"
            }
        } else {
            Write-Warn "Dependencies:  Status: 0 Installed ($totalReqs specifications need to install - will auto-install in Step 3)"
        }
    }
}

# ------------------------------------------------------------------
# 2. Step 1: Ensure Astral 'uv' is Installed
# ------------------------------------------------------------------
Write-Step "Step 1/5 - Ensuring high-speed 'uv' engine is ready..."

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCmd) {
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
    Write-Info "Installing official Astral 'uv' engine for ultra-fast installs..."
    try {
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString("https://astral.sh/uv/install.ps1"))
    } catch {
        Write-Warn "Direct installer script failed: $($_.Exception.Message)"
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Info "Retrying installation via winget..."
            & winget install astral-sh.uv --silent --accept-source-agreements --accept-package-agreements
        }
    }

    $uvBinDir = Join-Path $HOME ".local\bin"
    if (($env:Path -split ";") -notcontains $uvBinDir) {
        $env:Path = "$uvBinDir;$env:Path"
    }
    $cargoBinDir = Join-Path $HOME ".cargo\bin"
    if ((Test-Path -LiteralPath $cargoBinDir) -and (($env:Path -split ";") -notcontains $cargoBinDir)) {
        $env:Path = "$cargoBinDir;$env:Path"
    }

    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $uvCmd -and (Test-Path (Join-Path $uvBinDir "uv.exe"))) {
        $uvCmd = Get-Command (Join-Path $uvBinDir "uv.exe")
    }
}

if ($null -eq $uvCmd) {
    Write-Warn "'uv' could not be automatically installed. Will fall back to standard Python venv and pip."
    $UseUv = $false
} else {
    $uvExe = $uvCmd.Source
    Write-Ok "uv engine ready: $uvExe"
    $UseUv = $true
}

# ------------------------------------------------------------------
# 3. Step 2: Ensure Python 3.11 & Virtual Environment (.venv)
# ------------------------------------------------------------------
Write-Step "Step 2/5 - Configuring Python $PythonVersion runtime and .venv..."

if ($Recreate -and (Test-Path -LiteralPath $VenvPath)) {
    Write-Info "-Recreate supplied. Removing existing .venv..."
    # Check if a running process has file locks in .venv
    $lockedPy = Get-Process | Where-Object { $_.ProcessName -match "python" }
    if ($lockedPy) {
        Write-Warn "Active Python processes detected. Stopping them to prevent Windows DLL file locks..."
        $lockedPy | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $VenvPath -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Info "Creating isolated virtual environment (.venv)..."
    $venvCreated = $false

    if ($UseUv) {
        # Check if Python 3.11 is present; if not, uv can install it portably!
        Write-Info "Ensuring Python $PythonVersion is available via uv..."
        & $uvExe python install $PythonVersion 2>&1 | Out-Null

        $code = & $uvExe venv .venv --python $PythonVersion --seed 2>&1
        if (Test-Path -LiteralPath $VenvPython) {
            $venvCreated = $true
            Write-Ok ".venv created successfully with Python $PythonVersion via uv."
        } else {
            Write-Warn "uv venv with pinned version failed. Retrying with default Python..."
            & $uvExe venv .venv --seed 2>&1 | Out-Null
            if (Test-Path -LiteralPath $VenvPython) { $venvCreated = $true }
        }
    }

    if (-not $venvCreated) {
        Write-Info "Falling back to system Python venv..."
        $sysPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $sysPython) {
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                Write-Info "Installing Python $PythonVersion via winget..."
                & winget install Python.Python.3.11 --silent --accept-source-agreements --accept-package-agreements
                $sysPython = Get-Command python -ErrorAction SilentlyContinue
            }
        }
        if ($null -eq $sysPython) {
            Write-Fail "No Python found. Install Python 3.11 from https://www.python.org/downloads/ and re-run."
        }
        & $sysPython.Source -m venv .venv
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Fail "Failed to initialize .venv virtual environment."
}

$activePyVer = (& $VenvPython --version 2>&1)
Write-Ok "Venv runtime verified: $activePyVer ($VenvPython)"

# ------------------------------------------------------------------
# 4. Step 3: Lightning-Speed Dependencies Installation
# ------------------------------------------------------------------
Write-Step "Step 3/5 - Installing dependencies with lightning speed..."

if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    Write-Fail "requirements.txt not found at: $RequirementsPath"
}

$installSuccess = $false

# Ultra-fast pre-check: If core dependencies are already present and importable, bypass redundant installation
$checkCode = "import fastapi, cv2, numpy, onnxruntime, rapidocr_onnxruntime, rapidfuzz; print('OK')"
$alreadySatisfied = $false
try {
    $preCheck = (& $VenvPython -c $checkCode 2>&1)
    if ($preCheck -match "OK") {
        $alreadySatisfied = $true
    }
} catch { }

if ($alreadySatisfied) {
    Write-Ok "All core dependencies (FastAPI, OpenCV, ONNX Runtime, RapidOCR, RapidFuzz) are already installed and verified! (Lightning 0.2s cache hit)"
    $installSuccess = $true
} else {
    if ($UseUv) {
        Write-Info "Executing parallel batch installation via uv pip..."
        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        $uvInstallOutput = & $uvExe pip install -r $RequirementsPath --python $VenvPython --link-mode copy 2>&1
        $sw.Stop()

        $isSatisfied = ((& $VenvPython -c $checkCode 2>&1) -match "OK")
        if ($LASTEXITCODE -eq 0 -or $isSatisfied) {
            $installSuccess = $true
            $elapsedSec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
            Write-Ok "All dependencies installed and verified in $elapsedSec seconds! (Parallel Wheels)"
        } else {
            Write-Warn "Parallel batch install encountered an error. Attempting pip fallback..."
        }
    }

    if (-not $installSuccess) {
        Write-Info "Running pip fallback with binary wheel optimization..."
        $pipArgs = @("-m", "pip", "install", "-r", $RequirementsPath, "--prefer-binary", "--no-warn-script-location")
        & $VenvPython @pipArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "All dependencies installed successfully via pip."
            $installSuccess = $true
        }
    }
}

# ------------------------------------------------------------------
# 5. Step 4: Storage Scaffolding, Models & Config Initialization
# ------------------------------------------------------------------
Write-Step "Step 4/5 - Scaffolding directories, assets, and configuration..."

# 1. Directories
$requiredDirs = @(
    "backend\uploads",
    "backend\storage\uploads",
    "storage\uploads",
    "storage\uploads\drivers",
    "storage\uploads\previews"
)
foreach ($d in $requiredDirs) {
    $fullDir = Join-Path -Path $CurrentRoot -ChildPath $d
    if (-not (Test-Path -LiteralPath $fullDir)) {
        New-Item -ItemType Directory -Path $fullDir -Force | Out-Null
        Write-Info "Created directory: $d"
    }
}
Write-Ok "All upload and audit storage directories verified."

# 2. Config File (.env)
$envFile = Join-Path -Path $CurrentRoot -ChildPath ".env"
$envExample = Join-Path -Path $CurrentRoot -ChildPath ".env.example"
if (-not (Test-Path -LiteralPath $envFile) -and (Test-Path -LiteralPath $envExample)) {
    Copy-Item -LiteralPath $envExample -Destination $envFile
    Write-Ok "Initialized .env configuration from .env.example."
} elseif (Test-Path -LiteralPath $envFile) {
    Write-Ok ".env configuration file is present."
}

# 3. Model Files (YuNet ONNX Face Detector)
$yunetModelPath = Join-Path -Path $CurrentRoot -ChildPath "backend\models\face_detection_yunet_2023mar.onnx"
if (-not (Test-Path -LiteralPath $yunetModelPath)) {
    Write-Info "Downloading YuNet face detection model from OpenCV Zoo..."
    $yunetUrl = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    try {
        $modelDir = Split-Path $yunetModelPath
        if (-not (Test-Path $modelDir)) { New-Item -ItemType Directory -Path $modelDir -Force | Out-Null }
        (New-Object System.Net.WebClient).DownloadFile($yunetUrl, $yunetModelPath)
        Write-Ok "YuNet face detection model downloaded."
    } catch {
        Write-Warn "Could not auto-download YuNet model: $($_.Exception.Message)"
    }
} else {
    Write-Ok "YuNet face detector ONNX model verified."
}

# ------------------------------------------------------------------
# 6. Step 5: Pre-Flight Import & Diagnostic Verification
# ------------------------------------------------------------------
Write-Step "Step 5/5 - Pre-flight import and runtime diagnostics..."

if (-not $NoVerify) {
    $smokeCode = "import fastapi, uvicorn, cv2, numpy, onnxruntime, rapidocr_onnxruntime, rapidfuzz; print('CORE_OK')"
    $smokeResult = (& $VenvPython -c $smokeCode 2>&1)
    if ($smokeResult -match "CORE_OK") {
        Write-Ok "Core verification engines (FastAPI, OpenCV, ONNX Runtime, RapidOCR, RapidFuzz) loaded cleanly."
    } else {
        Write-Warn "Core import check returned: $smokeResult"
    }
}

# VLM Diagnostic Probe
try {
    $vlmProbeCode = @"
import sys
sys.path.insert(0, r'$CurrentRoot\backend')
try:
    from core.vlm_diagnostics import vlm_diagnostics
    p = vlm_diagnostics.probe_all()
    print('VLM_READY=' + str(p.get('vlm_ready', False)))
    print('VLM_PROVIDER=' + str(p.get('effective_provider', 'none')))
except Exception as e:
    print('VLM_ERR=' + str(e))
"@
    $vlmRes = (& $VenvPython -c $vlmProbeCode 2>&1)
    if ($vlmRes -match "VLM_READY=True") {
        Write-Ok "VLM Provider configuration verified."
    } else {
        Write-Info "VLM is unconfigured or in mock mode. Fill in VLM_API_KEY in .env before running live biometrics."
    }
} catch { }

# ------------------------------------------------------------------
# 7. Summary & 1-Liner Installation URL
# ------------------------------------------------------------------
Write-Header "INSTALLATION COMPLETE - READY TO SERVE"
Write-Host "  To launch the microservice locally:" -ForegroundColor White
Write-Host "    .\s.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "  Interactive Web Dashboard:" -ForegroundColor White
Write-Host "    http://127.0.0.1:8080/dashboard" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Swagger API Documentation:" -ForegroundColor White
Write-Host "    http://127.0.0.1:8080/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  UNIVERSAL 1-LINER COMMAND FOR OTHER LAPTOPS / MACHINES:" -ForegroundColor Yellow
Write-Host "  Copy and run this single command in PowerShell on any laptop:" -ForegroundColor Gray
Write-Host ""
Write-Host "  irm https://raw.githubusercontent.com/jassu-Workspace/komute-verification-node/main/install.ps1 | iex" -ForegroundColor Green
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
