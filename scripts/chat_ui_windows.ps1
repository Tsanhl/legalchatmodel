[CmdletBinding()]
param(
    [int]$UiPort = 8765,
    [int]$InferencePort = 8080,
    [string]$ModelPath = $env:LEGAL_GGUF_MODEL,
    [string]$LlamaServerPath = $env:LEGAL_LLAMA_SERVER,
    [ValidateSet("base", "v11-fused")]
    [string]$ModelProfile = "base",
    [int]$ContextSize = 32768
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Windows environment is missing. Run .\scripts\setup_windows.ps1 first."
}
if (-not $ModelPath) {
    throw "Set LEGAL_GGUF_MODEL or pass -ModelPath. See WINDOWS.md."
}
$ModelPath = (Resolve-Path $ModelPath).Path
if (-not $LlamaServerPath) {
    $FoundServer = Get-Command llama-server.exe -ErrorAction SilentlyContinue
    if ($FoundServer) { $LlamaServerPath = $FoundServer.Source }
}
if (-not $LlamaServerPath -or -not (Test-Path $LlamaServerPath -PathType Leaf)) {
    throw "Set LEGAL_LLAMA_SERVER to llama-server.exe or add it to PATH. See WINDOWS.md."
}
$LlamaServerPath = (Resolve-Path $LlamaServerPath).Path

if ($ModelProfile -eq "v11-fused") {
    $ManifestPath = "$ModelPath.v11.json"
    if (-not (Test-Path $ManifestPath -PathType Leaf)) {
        throw "V11 profile requires the export sidecar: $ManifestPath"
    }
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.profile -ne "v11-fused" -or -not $Manifest.gguf_sha256) {
        throw "The V11 sidecar is invalid. Re-export the model with scripts/export_v11_gguf.py."
    }
    $ActualHash = (Get-FileHash -Algorithm SHA256 $ModelPath).Hash.ToLowerInvariant()
    if ($ActualHash -ne ([string]$Manifest.gguf_sha256).ToLowerInvariant()) {
        throw "The V11 GGUF failed its SHA-256 integrity check."
    }
}

$ServerArgs = @(
    "-m", ('"' + $ModelPath + '"'),
    "--host", "127.0.0.1",
    "--port", [string]$InferencePort,
    "-c", [string]$ContextSize,
    "--parallel", "1",
    "--cors-origins", "localhost"
)
if ($env:LEGAL_LLAMA_GPU_LAYERS) {
    $ServerArgs += @("-ngl", $env:LEGAL_LLAMA_GPU_LAYERS)
}

Write-Host "Starting local llama-server..."
$PreviousLlamaApiKey = $env:LLAMA_API_KEY
$PreviousLegalApiKey = $env:LEGAL_LLAMA_API_KEY
$LocalApiKey = [guid]::NewGuid().ToString("N")
$env:LLAMA_API_KEY = $LocalApiKey
$env:LEGAL_LLAMA_API_KEY = $LocalApiKey
$HealthHeaders = @{ Authorization = "Bearer $LocalApiKey" }
$StartArguments = @{
    FilePath = $LlamaServerPath
    ArgumentList = $ServerArgs
    WorkingDirectory = $Root
    NoNewWindow = $true
    PassThru = $true
}
$LlamaProcess = Start-Process @StartArguments

try {
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 180; $Attempt++) {
        if ($LlamaProcess.HasExited) {
            throw "llama-server exited before becoming ready (code $($LlamaProcess.ExitCode))."
        }
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$InferencePort/health" -Headers $HealthHeaders -TimeoutSec 2
            if ($Health.status -eq "ok") { $Ready = $true; break }
        }
        catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $Ready) { throw "llama-server did not become ready within three minutes." }

    Write-Host "Opening LegalChatModel at http://127.0.0.1:$UiPort/" -ForegroundColor Green
    Start-Process "http://127.0.0.1:$UiPort/"
    $UiArguments = @(
        (Join-Path $Root "legal_chat_ui\server.py"),
        "--backend", "llama-server",
        "--llama-base-url", "http://127.0.0.1:$InferencePort/v1",
        "--llama-model-profile", $ModelProfile,
        "--host", "127.0.0.1",
        "--port", [string]$UiPort
    )
    & $Python @UiArguments
}
finally {
    if ($LlamaProcess -and -not $LlamaProcess.HasExited) {
        Stop-Process -Id $LlamaProcess.Id -Force
    }
    $env:LLAMA_API_KEY = $PreviousLlamaApiKey
    $env:LEGAL_LLAMA_API_KEY = $PreviousLegalApiKey
}
