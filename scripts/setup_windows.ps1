<#
.SYNOPSIS
    Build and install DeepSpeed (with the async_io / DeepNVMe op) on Windows.

.DESCRIPTION
    Installing DeepSpeed on Windows is the hard part of running models larger
    than VRAM. This script automates the fiddly bits:

      1. Locates the Visual Studio C++ toolchain via vswhere and imports the
         MSVC developer environment (vcvars64.bat) into this PowerShell session
         so cl.exe / link.exe are on PATH.
      2. Verifies a CUDA toolkit (nvcc) is reachable.
      3. Sets the environment variables DeepSpeed's Windows build expects
         (DISTUTILS_USE_SDK=1) and, with -BuildAio, DS_BUILD_AIO=1 so the
         async_io op used for NVMe offloading is compiled in.
      4. Builds DeepSpeed from source (clone or pip) and installs the wheel.
      5. Runs ds_report to confirm which ops are available.

    Run from an elevated PowerShell if you also need it to install toolchains.

.PARAMETER BuildAio
    Compile the async_io op (DeepNVMe). Required for NVMe offloading.

.PARAMETER Ref
    DeepSpeed git ref/tag to build (default: master).

.PARAMETER Python
    Python executable to use (default: the one on PATH).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -BuildAio
#>
[CmdletBinding()]
param(
    [switch]$BuildAio,
    [string]$Ref = "master",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[error] $msg" -ForegroundColor Red; exit 1 }

# --- 1. Locate and import the MSVC build environment ----------------------
Write-Step "Locating Visual C++ build tools (vswhere)"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Fail "vswhere.exe not found. Install 'Build Tools for Visual Studio' with the 'Desktop development with C++' workload."
}

$vsPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsPath) {
    Fail "No VS installation with the C++ toolset found. Install the 'Desktop development with C++' workload."
}
Write-Host "Found Visual Studio at: $vsPath"

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { Fail "vcvars64.bat not found under $vsPath" }

Write-Step "Importing MSVC x64 developer environment"
# Run vcvars64.bat in a child cmd, then import the resulting env vars here.
$tmp = [System.IO.Path]::GetTempFileName()
cmd /c "`"$vcvars`" && set > `"$tmp`""
Get-Content $tmp | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}
Remove-Item $tmp -ErrorAction SilentlyContinue

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Fail "cl.exe still not on PATH after importing vcvars64. Cannot continue."
}
Write-Host "cl.exe is available."

# --- 2. Verify CUDA toolkit ----------------------------------------------
Write-Step "Checking CUDA toolkit (nvcc)"
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    nvcc --version | Select-Object -Last 2 | ForEach-Object { Write-Host $_ }
} else {
    Write-Warn "nvcc not found on PATH. DeepSpeed CUDA ops need the CUDA Toolkit. Set CUDA_HOME/PATH and re-run if the build fails."
}

# --- 3. Build flags -------------------------------------------------------
Write-Step "Configuring DeepSpeed build flags"
$env:DISTUTILS_USE_SDK = "1"
# Most ops are JIT-built on first use; we precompile only what we need to keep
# the build robust on Windows. async_io powers DeepNVMe NVMe offloading.
if ($BuildAio) {
    $env:DS_BUILD_AIO = "1"
    Write-Host "DS_BUILD_AIO=1 (async_io / DeepNVMe will be compiled)"
} else {
    Write-Warn "Building without async_io. NVMe offload will be unavailable; pass -BuildAio to enable it."
}
Write-Host "DISTUTILS_USE_SDK=1"

& $Python -m pip install --upgrade pip wheel setuptools ninja | Out-Host

# --- 4. Build & install DeepSpeed ----------------------------------------
Write-Step "Building DeepSpeed from source (ref: $Ref)"
$work = Join-Path $env:TEMP "plvram-deepspeed-build"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
git clone --depth 1 --branch $Ref https://github.com/microsoft/DeepSpeed.git $work 2>$null
if (-not (Test-Path $work)) {
    Write-Warn "Branch '$Ref' clone failed; cloning default branch."
    git clone --depth 1 https://github.com/microsoft/DeepSpeed.git $work
}

Push-Location $work
try {
    & $Python -m pip install . --no-build-isolation | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "DeepSpeed build/install failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}

# --- 5. Verify ------------------------------------------------------------
Write-Step "Verifying installation (ds_report)"
& $Python -m deepspeed.env_report 2>$null
if ($LASTEXITCODE -ne 0) {
    try { ds_report } catch { Write-Warn "ds_report not runnable; import check follows." }
}
& $Python -c "import deepspeed; print('DeepSpeed', deepspeed.__version__, 'imported OK')"

Write-Host "`nDone. If async_io shows [OKAY] in ds_report, NVMe offloading is ready." -ForegroundColor Green
