<#
.SYNOPSIS
    Run Shorts Miner locally and publish it through a Cloudflare Tunnel.

.DESCRIPTION
    Starts Streamlit on localhost and exposes it with cloudflared, so visitors
    reach the app while YouTube sees this machine's residential IP instead of a
    blocked datacenter one. Ctrl+C stops both.

    Without a named tunnel this uses a free quick tunnel: no account needed,
    but the https://....trycloudflare.com URL changes on every start. Once a
    domain is on your Cloudflare account, run with -Setup once to create a
    named tunnel with a fixed hostname, and every later start reuses it.

.EXAMPLE
    .\run_public.ps1
    .\run_public.ps1 -Setup -Hostname shorts.example.com
    .\run_public.ps1 -KeepProxy
#>
param(
    # One-time: log in to Cloudflare, create the named tunnel, and route
    # -Hostname to it. Needs a domain on your Cloudflare account.
    [switch]$Setup,
    [string]$Hostname,
    [string]$TunnelName = "shorts-miner",
    [int]$Port = 8501,
    # By default PROXY_URL is cleared for this run: the point of hosting from
    # home is YouTube seeing your own IP, and a proxy would hide it.
    [switch]$KeepProxy
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared is not installed. Run: winget install --id Cloudflare.cloudflared (then open a new terminal)."
}

$hasNamedTunnel = Test-Path "$HOME\.cloudflared\$TunnelName.yml"

if ($Setup) {
    if (-not $Hostname) { throw "-Setup needs -Hostname, e.g. -Hostname shorts.example.com" }
    if (-not (Test-Path "$HOME\.cloudflared\cert.pem")) {
        Write-Host "Opening the browser to log in to Cloudflare. Pick the domain for $Hostname."
        cloudflared tunnel login
    }
    $existing = cloudflared tunnel list --output json | ConvertFrom-Json | Where-Object { $_.name -eq $TunnelName }
    if (-not $existing) { cloudflared tunnel create $TunnelName | Out-Host }
    $tunnelId = (cloudflared tunnel list --output json | ConvertFrom-Json | Where-Object { $_.name -eq $TunnelName }).id
    cloudflared tunnel route dns --overwrite-dns $TunnelName $Hostname | Out-Host
    @"
tunnel: $tunnelId
credentials-file: $HOME\.cloudflared\$tunnelId.json
ingress:
  - hostname: $Hostname
    service: http://localhost:$Port
  - service: http_status:404
"@ | Set-Content -Encoding utf8 "$HOME\.cloudflared\$TunnelName.yml"
    Write-Host "Named tunnel ready: https://$Hostname"
    $hasNamedTunnel = $true
}

# Use whichever Python actually has the project's dependencies installed.
# Candidates are strings split on spaces: PowerShell flattens nested arrays,
# and a bare "py" with no args opens an interactive REPL that hangs forever.
$pyExe = $null
$pyArgs = @()
foreach ($candidate in "py -3.14", "python") {
    $parts = $candidate -split " "
    if (-not (Get-Command $parts[0] -ErrorAction SilentlyContinue)) { continue }
    $candidateArgs = @($parts | Select-Object -Skip 1)
    # EAP=Stop would turn any stderr from a failed import into a terminating error.
    $ErrorActionPreference = "Continue"
    & $parts[0] @candidateArgs -c "import streamlit" *> $null
    $ok = $LASTEXITCODE -eq 0
    $ErrorActionPreference = "Stop"
    if ($ok) { $pyExe = $parts[0]; $pyArgs = $candidateArgs; break }
}
if (-not $pyExe) { throw "No Python with streamlit installed. Run: pip install -r requirements.txt" }

function Test-PortOpen([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.ConnectAsync("127.0.0.1", $p).Wait(500) -and $client.Connected } catch { $false } finally { $client.Dispose() }
}

if (-not $KeepProxy) { $env:PROXY_URL = "" }

$streamlit = Start-Process -FilePath $pyExe -NoNewWindow -PassThru -ArgumentList (
    $pyArgs + @("-m", "streamlit", "run", "app.py", "--server.port", "$Port", "--server.headless", "true")
)

try {
    Write-Host "Waiting for Streamlit on port $Port..."
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-PortOpen $Port)) {
        if ($streamlit.HasExited) { throw "Streamlit exited early (code $($streamlit.ExitCode))." }
        if ((Get-Date) -gt $deadline) { throw "Streamlit didn't start within 60s." }
        Start-Sleep -Seconds 1
    }

    # cloudflared logs to stderr; under EAP=Stop that aborts when output is redirected.
    $ErrorActionPreference = "Continue"
    if ($hasNamedTunnel) {
        Write-Host "Starting named tunnel '$TunnelName' (fixed URL). Ctrl+C to stop."
        cloudflared tunnel --config "$HOME\.cloudflared\$TunnelName.yml" run $TunnelName
    } else {
        Write-Host "Starting quick tunnel. Look for the https://....trycloudflare.com URL below. Ctrl+C to stop."
        cloudflared tunnel --url "http://localhost:$Port"
    }
} finally {
    if (-not $streamlit.HasExited) { Stop-Process -Id $streamlit.Id -Force }
}
