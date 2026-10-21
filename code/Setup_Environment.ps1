$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$applicationDataRoot = Join-Path (Split-Path -Parent $projectRoot) 'data'
if ($env:TRO_DATA_ROOT) { $applicationDataRoot = [IO.Path]::GetFullPath($env:TRO_DATA_ROOT) }
$localEnvironment = Join-Path $applicationDataRoot '.python310'
$localPython = Join-Path $localEnvironment 'python.exe'
$pythonArchive = Join-Path $applicationDataRoot "runtime\python-3.10.11-embed-amd64.zip"
$getPip = Join-Path $applicationDataRoot "runtime\get-pip.py"
$requirements = Join-Path $projectRoot "requirements-runtime.txt"
$wheelhouse = Join-Path $applicationDataRoot "runtime\wheelhouse"

function Test-Python310([string]$executable) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { return $false }
    try {
        $version = & $executable -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return ($LASTEXITCODE -eq 0 -and "$version".Trim() -eq "3.10")
    } catch { return $false }
}

if (-not (Test-Python310 $localPython)) {
    if (Test-Path -LiteralPath $localEnvironment) {
        throw "The local .python310 directory exists but is incomplete. Remove it and run setup again."
    }
    if (-not (Test-Path -LiteralPath $pythonArchive -PathType Leaf)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $pythonArchive) -Force | Out-Null
        Write-Host "Downloading Python 3.10 runtime..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip" -OutFile $pythonArchive
    }
    Write-Host "Extracting bundled Python 3.10 runtime..." -ForegroundColor Cyan
    Expand-Archive -LiteralPath $pythonArchive -DestinationPath $localEnvironment
    if (-not (Test-Python310 $localPython)) {
        throw "The bundled Python 3.10 runtime could not be extracted."
    }

    $pth = Join-Path $localEnvironment "python310._pth"
    if (-not (Test-Path -LiteralPath $pth -PathType Leaf)) {
        throw "Bundled Python path configuration is missing: $pth"
    }
    $pthLines = @(Get-Content -LiteralPath $pth)
    $pthLines = @($pthLines | Where-Object { $_ -ne "#import site" -and $_ -ne "import site" -and $_ -ne "Lib\site-packages" })
    $pthLines += "Lib\site-packages"
    $pthLines += "import site"
    [System.IO.File]::WriteAllLines($pth, [string[]]$pthLines, [System.Text.Encoding]::ASCII)
}

Write-Host "Installing verified dependencies..." -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $getPip -PathType Leaf)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $getPip) -Force | Out-Null
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
}
$offlineWheels = @(Get-ChildItem -LiteralPath $wheelhouse -Filter *.whl -File -ErrorAction SilentlyContinue)
if ($offlineWheels.Count -gt 0) {
    & $localPython $getPip --no-index --find-links $wheelhouse "pip==25.3" "setuptools==84.0.0" "wheel==0.45.1"
    if ($LASTEXITCODE -ne 0) { throw "Could not initialize bundled pip." }
    & $localPython -m pip install --disable-pip-version-check --no-index --find-links $wheelhouse -r $requirements
} else {
    & $localPython $getPip
    if ($LASTEXITCODE -ne 0) { throw "Could not initialize pip." }
    & $localPython -m pip install --disable-pip-version-check -r $requirements
}
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

$env:PYTHONDONTWRITEBYTECODE = "1"
& $localPython (Join-Path $projectRoot "portable_check.py")
if ($LASTEXITCODE -ne 0) { throw "Portable environment validation failed." }
