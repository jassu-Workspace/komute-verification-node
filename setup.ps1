<#
.SYNOPSIS
  Installs and checks everything Komute-Local needs, from a fresh Windows machine to a ready node.

.DESCRIPTION
  Built for a fresh machine that may have none of the prerequisites.
  0. Platform (x64, Windows 10 1809 / Server 2019+, RAM, disk), network reach to GitHub and PyPI,
     and the Microsoft Visual C++ runtime (downloaded, Microsoft signature checked, installed; asks for admin once).
  1. uv (winget, else the official installer from astral.sh).
  2. Python 3.12 and the virtual environment in %LOCALAPPDATA%\komute\venv (outside OneDrive).
  3. The exact dependency set from requirements.lock, then an import check of every runtime module.
  4. The four ONNX models. Each download is checked against backend\models\models.lock (SHA-256)
     before it is put in place. A file that does not match is deleted, never used.
  5. Dev keys (once) and the storage folder.
  6. A final check: model hashes, settings, and the unit tests.

  Safe to run again: every step skips what is already correct. Start the app afterwards with .\run.ps1.

.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -SkipTests
  .\setup.ps1 -Force        # re-download every model and reinstall every package
  powershell -ExecutionPolicy Bypass -File .\setup.ps1    # if script execution is blocked
#>
param(
    [switch]$SkipTests,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Windows PowerShell download progress bars are very slow
Set-Location -LiteralPath $PSScriptRoot

$Home_ = Join-Path $env:LOCALAPPDATA 'komute'
$VenvDir = Join-Path $Home_ 'venv'
$Py = Join-Path $VenvDir 'Scripts\python.exe'
$DevEnv = Join-Path $Home_ 'dev.env'
$StorageDir = Join-Path $Home_ 'storage'
$LockReqs = Join-Path $PSScriptRoot 'requirements.lock'
$ModelsDir = Join-Path $PSScriptRoot 'backend\models'
$LockFile = Join-Path $ModelsDir 'models.lock'

$RuntimeImports = @('fastapi', 'uvicorn', 'pydantic', 'pydantic_settings', 'multipart', 'psutil', 'cv2',
                    'numpy', 'onnxruntime', 'zxingcpp', 'rapidfuzz', 'scipy', 'cryptography', 'tzdata')

# Public sources of the locked model files. The SHA-256 in models.lock is the authority, not the URL.
$ModelSources = @{
    'face_detection_yunet_2023mar.onnx'         = @{ Url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' }
    'yolo-v9-t-384-license-plates-end2end.onnx' = @{ Url = 'https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx' }
    'cct_xs_v2_global.onnx'                     = @{ Url = 'https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global.onnx' }
    'w600k_r50.onnx'                            = @{ Url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'; ZipEntry = 'w600k_r50.onnx' }
}

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    ok  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    !!  $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "`nERROR: $text" -ForegroundColor Red; exit 1 }

function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

# Old .NET defaults in Windows PowerShell 5.1 can offer TLS 1.0 only; GitHub and astral.sh need TLS 1.2+.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$HasCurl = [bool](Get-Command curl.exe -ErrorAction SilentlyContinue)

function Save-Url([string]$Url, [string]$OutFile) {
    # curl.exe (Windows 10 1803+) is fast and honours HTTPS_PROXY; Invoke-WebRequest is the fallback. TLS checks stay on.
    if ($HasCurl) {
        & curl.exe --fail --location --silent --show-error --retry 3 --connect-timeout 20 --output $OutFile $Url
        if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE): $Url" }
    } else {
        for ($i = 1; $i -le 3; $i++) {
            try { Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -TimeoutSec 600; return }
            catch { if ($i -eq 3) { throw "download failed: $Url ($($_.Exception.Message))" }; Start-Sleep -Seconds (2 * $i) }
        }
    }
}

function Test-Admin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-VcRedist {
    $k = 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
    $reg = Get-ItemProperty -Path $k -ErrorAction SilentlyContinue
    if ($reg -and $reg.Installed -eq 1 -and $reg.Major -ge 14 -and $reg.Minor -ge 30) { return $true }
    $sys = Join-Path $env:WINDIR 'System32'
    return (Test-Path (Join-Path $sys 'msvcp140.dll')) -and (Test-Path (Join-Path $sys 'vcruntime140_1.dll'))
}

# ---------------------------------------------------------------- 0. platform
Step 'Checking the platform'
if ($env:OS -ne 'Windows_NT') { Fail 'setup.ps1 is for Windows. On Ubuntu use: sudo ./setup.sh' }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
    Fail "An x64 (AMD64) machine is required; this one is $env:PROCESSOR_ARCHITECTURE. ONNX Runtime and OpenCV wheels are x64."
}
if ($PSVersionTable.PSVersion -lt [Version]'5.1') { Fail "PowerShell 5.1 or later is required (found $($PSVersionTable.PSVersion))." }
$build = [Environment]::OSVersion.Version.Build
if ($build -lt 17763) { Fail "Windows 10 1809 / Server 2019 (build 17763) or later is required; this is build $build." }
if (-not $HasCurl) { Warn 'curl.exe not found; downloads use Invoke-WebRequest (slower).' }
if ($PSScriptRoot -match 'OneDrive') { Warn 'The repo is inside OneDrive. That is fine for code; the venv and storage go to %LOCALAPPDATA%.' }
$free = (Get-PSDrive -Name ($env:LOCALAPPDATA.Substring(0, 1))).Free
if ($free -lt 3GB) { Fail ("Less than 3 GB free on the drive of $env:LOCALAPPDATA ({0:N1} GB)." -f ($free / 1GB)) }
$ramGb = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
if ($ramGb -lt 2) { Fail ("At least 2 GB RAM is required (one verification peaks near 1.2 GB); found {0:N1} GB." -f $ramGb) }
Ok ("Windows build {0}, x64, PowerShell {1}, {2:N1} GB RAM, {3:N1} GB free" -f $build, $PSVersionTable.PSVersion, $ramGb, ($free / 1GB))

Step 'Checking network access'
foreach ($h in @('https://github.com', 'https://pypi.org/simple/', 'https://files.pythonhosted.org')) {
    try { Invoke-WebRequest -Uri $h -Method Head -UseBasicParsing -TimeoutSec 20 | Out-Null }
    catch {
        # Some hosts refuse HEAD; any HTTP answer still proves the network path works.
        if (-not $_.Exception.Response) { Fail "Cannot reach $h. Check the internet connection or set HTTPS_PROXY. ($($_.Exception.Message))" }
    }
}
Ok 'github.com and PyPI are reachable'

# ---------------------------------------------------------------- 0b. Visual C++ runtime
Step 'Checking the Microsoft Visual C++ runtime (needed by ONNX Runtime and OpenCV)'
if (Test-VcRedist) {
    Ok 'Visual C++ 2015-2022 x64 runtime installed'
} else {
    $vc = Join-Path $env:TEMP ('vc_redist.x64-' + [guid]::NewGuid().ToString('N') + '.exe')
    try {
        Warn 'Visual C++ runtime missing. Downloading the official Microsoft installer.'
        Save-Url 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $vc
        $sig = Get-AuthenticodeSignature -LiteralPath $vc
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
            throw "vc_redist.x64.exe signature check failed ($($sig.Status)). The file was discarded."
        }
        $vcArgs = @('/install', '/quiet', '/norestart')
        $p = if (Test-Admin) { Start-Process -FilePath $vc -ArgumentList $vcArgs -Wait -PassThru }
             else { Warn 'Administrator rights needed: approve the Windows prompt.'; Start-Process -FilePath $vc -ArgumentList $vcArgs -Verb RunAs -Wait -PassThru }
        if ($p.ExitCode -notin 0, 1638, 3010) { throw "vc_redist.x64.exe failed with exit code $($p.ExitCode)." }  # 1638 = newer present, 3010 = reboot pending
        if ($p.ExitCode -eq 3010) { Warn 'Visual C++ runtime installed; Windows wants a restart. Restart before the first real run.' }
        Ok 'Visual C++ runtime installed (Microsoft signature verified)'
    } catch {
        Fail ("Could not install the Visual C++ runtime: $($_.Exception.Message)`n" +
              "Install it by hand from https://aka.ms/vs/17/release/vc_redist.x64.exe and run again.")
    } finally {
        Remove-Item -LiteralPath $vc -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- 1. uv
Step 'Installing uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Warn 'uv not found. Installing it with winget.'
        & winget install --id astral-sh.uv --exact --silent --accept-package-agreements --accept-source-agreements | Out-Host
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Warn 'Installing uv with the official installer from astral.sh.'
        powershell -NoProfile -ExecutionPolicy ByPass -Command ('[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072; ' +
                                                                'irm https://astral.sh/uv/install.ps1 | iex') | Out-Host
    }
    # A fresh install is not on this session's PATH yet.
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                (Join-Path $env:USERPROFILE '.local\bin')
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail 'uv could not be installed. Install it by hand: https://docs.astral.sh/uv/' }
}
Ok ("uv " + ((& uv --version) -replace '^uv ', ''))

# ---------------------------------------------------------------- 2. Python 3.12 and venv
Step 'Installing Python 3.12 and the virtual environment'
& uv python install 3.12
if ($LASTEXITCODE -ne 0) { Fail 'uv could not install Python 3.12.' }
$venvOk = $false
if (Test-Path -LiteralPath $Py) {
    $venvOk = ((& $Py -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null) -eq '3.12')
}
if (-not $venvOk) {
    if (Test-Path -LiteralPath $VenvDir) { Warn "The venv at $VenvDir is not Python 3.12. Recreating it." }
    & uv venv --python 3.12 --clear $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Py)) { Fail 'uv could not create the virtual environment.' }
}
Ok ("Python " + (& $Py -c "import sys; print(sys.version.split()[0])") + "  ($VenvDir)")

# ---------------------------------------------------------------- 3. dependencies
Step 'Installing dependencies (requirements.lock, exact versions)'
if (-not (Test-Path -LiteralPath $LockReqs)) { Fail "requirements.lock not found at $LockReqs" }
# install, not sync: pins every locked package without removing extra tools someone added to the venv.
$syncArgs = @('pip', 'install', '--python', $Py, '-r', $LockReqs)
if ($Force) { $syncArgs += '--reinstall' }
& uv @syncArgs
if ($LASTEXITCODE -ne 0) { Fail 'Dependency install failed. Fix the error above and run again.' }
$code = "import importlib.util, sys; bad = [m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]; print(' '.join(bad)); sys.exit(1 if bad else 0)"
$missing = & $Py -c $code @RuntimeImports 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { Fail "Runtime modules still missing: $($missing.Trim())" }
& $Py -c 'import backend.core' 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail 'The backend package does not import. Run: python -c "import backend.core"' }
Ok "all $($RuntimeImports.Count) runtime modules import; backend guards active"

# ---------------------------------------------------------------- 4. models
Step 'Installing model files (verified against models.lock)'
if (-not (Test-Path -LiteralPath $LockFile)) { Fail "models.lock not found at $LockFile" }
$lock = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json
foreach ($name in @($lock.PSObject.Properties.Name)) {
    $want = $lock.$name.ToLowerInvariant()
    $dest = Join-Path $ModelsDir $name
    if (-not $Force -and (Test-Path -LiteralPath $dest) -and (Get-Sha256 $dest) -eq $want) { Ok "$name (present, hash matches)"; continue }
    if (Test-Path -LiteralPath $dest) { Warn "$name does not match models.lock. Replacing it." }
    $src = $ModelSources[$name]
    if ($null -eq $src) { Fail "$name is in models.lock but has no download source. Copy the licensed file to $ModelsDir." }
    $part = "$dest.part"
    $zip = Join-Path $env:TEMP ("komute-" + [guid]::NewGuid().ToString('N') + '.zip')
    try {
        Write-Host "    ..  downloading $name" -ForegroundColor DarkGray
        if ($src.ZipEntry) {
            Save-Url $src.Url $zip
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $archive = [IO.Compression.ZipFile]::OpenRead($zip)
            try {
                $entry = $archive.Entries | Where-Object { $_.FullName -eq $src.ZipEntry -or $_.FullName.EndsWith('/' + $src.ZipEntry) } | Select-Object -First 1
                if ($null -eq $entry) { throw "$($src.ZipEntry) is not inside $($src.Url)" }
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $part, $true)
            } finally { $archive.Dispose() }
        } else {
            Save-Url $src.Url $part
        }
        $got = Get-Sha256 $part
        if ($got -ne $want) { throw "$name hash mismatch: expected $want, got $got. The file was discarded." }
        Move-Item -LiteralPath $part -Destination $dest -Force
        Ok "$name (downloaded, hash matches)"
    } catch {
        Fail $_.Exception.Message
    } finally {
        Remove-Item -LiteralPath $part, $zip -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- 5. keys and storage
Step 'Preparing keys and storage'
$HostKeys = [bool]($env:KOMUTE_STORAGE_KEY_B64 -and $env:KOMUTE_MANIFEST_KEY_B64)
if ($HostKeys) {
    Ok 'keys injected by the host environment (KOMUTE_*_KEY_B64); no dev keys created'
} elseif (Test-Path -LiteralPath $DevEnv) {
    Ok "dev keys exist ($DevEnv, values not shown)"
} else {
    & $Py (Join-Path $PSScriptRoot 'scripts\make_dev_keys.py')
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $DevEnv)) { Fail 'Could not create dev keys.' }
    Ok "dev keys created ($DevEnv, values not shown)"
}
if ($env:KOMUTE_STORAGE_ROOT) { $StorageDir = $env:KOMUTE_STORAGE_ROOT }
New-Item -ItemType Directory -Force -Path $StorageDir | Out-Null
Ok "storage folder: $StorageDir"
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.env')) {
    Ok '.env found (git-ignored); its values apply unless an environment variable overrides them'
} else {
    Ok 'no .env (defaults apply). To tune thresholds: copy .env.example to .env and edit'
}

# ---------------------------------------------------------------- 6. final check
Step 'Final check'
$saved = @{}
Get-ChildItem Env: | Where-Object { $_.Name -like 'KOMUTE_*' } | ForEach-Object { $saved[$_.Name] = $_.Value }
if (-not $HostKeys) {
    foreach ($line in Get-Content -LiteralPath $DevEnv) {
        if ($line -match '^\s*([A-Z0-9_]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
}
$env:PYTHONUTF8 = '1'
if (-not $env:KOMUTE_ENV) { $env:KOMUTE_ENV = 'dev' }
$env:KOMUTE_STORAGE_ROOT = $StorageDir
if (-not $env:KOMUTE_LIVENESS_MODEL -and $env:KOMUTE_ENV -eq 'dev') { $env:KOMUTE_DEV_SKIP_LIVENESS = 'true' }
$check = "from pathlib import Path; from backend.core.registry import verify_lock; from backend.core.config import Settings; " +
         "verify_lock(Path('backend/models'), Path('backend/models/models.lock')); Settings(); print('ok')"
$out = & $Py -c $check 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { Fail ("Model lock or settings check failed:`n" + $out.Trim()) }
Ok 'model hashes and settings valid'

if ($SkipTests) {
    Warn 'tests skipped (-SkipTests)'
} else {
    Write-Host '    ..  running unit and API tests (about 30-60 s)' -ForegroundColor DarkGray
    # Tests build their own settings: hide every KOMUTE_* variable from them, then restore.
    Get-ChildItem Env: | Where-Object { $_.Name -like 'KOMUTE_*' } | ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
    $out = & $Py -m pytest -m 'not samples' -q -p no:cacheprovider --basetemp (Join-Path $Home_ 'pytest-tmp') 2>&1 | Out-String
    $saved.GetEnumerator() | ForEach-Object { [Environment]::SetEnvironmentVariable($_.Key, $_.Value, 'Process') }
    if ($LASTEXITCODE -ne 0) { Write-Host $out; Fail 'Tests failed. See the output above.' }
    Ok (($out.Trim() -split "`n")[-1].Trim())
}

if (-not $env:KOMUTE_LIVENESS_MODEL) { Warn 'No liveness model: dev runs skip liveness. Production needs KOMUTE_LIVENESS_MODEL.' }
Write-Host "`nSetup complete. Start the app with: .\run.ps1" -ForegroundColor Green
