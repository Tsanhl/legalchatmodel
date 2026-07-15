[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"
    & py -3 -m venv .venv
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"
    & python -m venv .venv
}
else {
    throw "Python 3.10 or newer was not found. Install Python, then run this script again."
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirements-windows.txt")

Write-Host "Windows application environment is ready." -ForegroundColor Green
Write-Host "Next: follow WINDOWS.md to install llama.cpp and choose a GGUF model."
