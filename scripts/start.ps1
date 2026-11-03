[CmdletBinding()]
param(
    [switch]$Production
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendDirectory = Join-Path $repositoryRoot "backend"
$frontendDirectory = Join-Path $repositoryRoot "frontend"
$venvDirectory = Join-Path $backendDirectory ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"
$activateScript = Join-Path $venvDirectory "Scripts\Activate.ps1"
$runtimeDirectory = Join-Path $repositoryRoot ".runtime"
$logDirectory = Join-Path $runtimeDirectory "logs"
$processFile = Join-Path $runtimeDirectory "processes.json"

function Import-ProjectEnvironment {
    $envFile = Join-Path $repositoryRoot ".env"
    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $envFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $key, $value = $trimmed -split "=", 2
        if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            throw "Invalid environment variable name in .env: $key"
        }
        $value = $value.Trim()
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not (Test-Path "Env:$key") -or [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($key))) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }
}

function Stop-LaunchedProcess([int]$ProcessId) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) was not found. Install Python 3.10 and ensure py.exe is on PATH."
}
$pythonVersion = & py -3.10 --version 2>&1
if ($LASTEXITCODE -ne 0 -or $pythonVersion -notmatch '^Python 3\.10\.') {
    throw "Python 3.10 is required. Detected: $pythonVersion"
}
if (-not (Get-Command node -ErrorAction SilentlyContinue) -or
    -not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js and npm are required. Install a supported Node.js LTS release and reopen PowerShell."
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf) -or
    -not (Test-Path -LiteralPath $activateScript -PathType Leaf)) {
    throw "Backend virtual environment is missing. Follow the Python setup steps in README.md first."
}
$venvVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $venvVersion.Trim() -ne "3.10") {
    throw "backend/.venv must use Python 3.10; found $venvVersion"
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory "node_modules\next\dist\bin\next"))) {
    throw "Frontend dependencies are missing. Run npm ci in frontend/ as described in README.md."
}

Import-ProjectEnvironment
if (-not $env:API_PORT) { $env:API_PORT = "8000" }
if (-not $env:FRONTEND_PORT) { $env:FRONTEND_PORT = "3000" }
if (-not $env:NEXT_PUBLIC_API_URL) { $env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:$($env:API_PORT)" }
if (-not $env:PAPER2PRODUCT_CORS_ORIGINS) { $env:PAPER2PRODUCT_CORS_ORIGINS = "http://localhost:$($env:FRONTEND_PORT),http://127.0.0.1:$($env:FRONTEND_PORT)" }

$apiPort = 0
$frontendPort = 0
if (-not [int]::TryParse($env:API_PORT, [ref]$apiPort) -or $apiPort -lt 1 -or $apiPort -gt 65535) {
    throw "API_PORT must be an integer from 1 through 65535."
}
if (-not [int]::TryParse($env:FRONTEND_PORT, [ref]$frontendPort) -or $frontendPort -lt 1 -or $frontendPort -gt 65535) {
    throw "FRONTEND_PORT must be an integer from 1 through 65535."
}
$apiBaseUrl = "http://127.0.0.1:$apiPort"
$healthUrl = "$apiBaseUrl/health"
$frontendUrl = "http://localhost:$frontendPort"

if (Test-Path -LiteralPath $processFile -PathType Leaf) {
    $oldProcesses = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $running = @($oldProcesses.processes | Where-Object { Get-Process -Id $_.pid -ErrorAction SilentlyContinue })
    if ($running.Count -gt 0) {
        throw "PAPER2PRODUCT processes are already recorded as running. Run scripts/stop.ps1 first."
    }
}

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
& $activateScript
$pythonExecutable = (Resolve-Path $venvPython).Path
$nodeExecutable = (Get-Command node).Source
$nextCli = Join-Path $frontendDirectory "node_modules\next\dist\bin\next"
$startedProcesses = [System.Collections.Generic.List[object]]::new()

try {
    if ($Production) {
        Write-Host "Building frontend production bundle..."
        Push-Location $frontendDirectory
        try {
            & npm run build
            if ($LASTEXITCODE -ne 0) {
                throw "Frontend production build failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
    }

    $apiProcess = Start-Process -FilePath $pythonExecutable `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$apiPort") `
        -WorkingDirectory $backendDirectory -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDirectory "api.stdout.log") `
        -RedirectStandardError (Join-Path $logDirectory "api.stderr.log")
    $startedProcesses.Add([PSCustomObject]@{ name = "api"; pid = $apiProcess.Id })

    $frontendMode = if ($Production) { "start" } else { "dev" }
    $frontendProcess = Start-Process -FilePath $nodeExecutable `
        -ArgumentList @($nextCli, $frontendMode, "--hostname", "127.0.0.1", "--port", "$frontendPort") `
        -WorkingDirectory $frontendDirectory -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDirectory "frontend.stdout.log") `
        -RedirectStandardError (Join-Path $logDirectory "frontend.stderr.log")
    $startedProcesses.Add([PSCustomObject]@{ name = "frontend"; pid = $frontendProcess.Id })

    $deadline = (Get-Date).AddSeconds(90)
    $healthResponse = $null
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $apiProcess.Id -ErrorAction SilentlyContinue)) {
            throw "API exited before becoming healthy. Review $logDirectory\api.stderr.log."
        }
        try {
            $healthResponse = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3
            if ($healthResponse.StatusCode -eq 200) { break }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    if ($null -eq $healthResponse -or $healthResponse.StatusCode -ne 200) {
        throw "API did not return HTTP 200 from $healthUrl within 90 seconds."
    }

    $frontendReady = $false
    $frontendDeadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $frontendDeadline) {
        if (-not (Get-Process -Id $frontendProcess.Id -ErrorAction SilentlyContinue)) {
            throw "Frontend exited before becoming available. Review $logDirectory\frontend.stderr.log."
        }
        try {
            $page = Invoke-WebRequest -Uri $frontendUrl -UseBasicParsing -TimeoutSec 5
            if ($page.StatusCode -eq 200) {
                $frontendReady = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $frontendReady) {
        throw "Frontend did not return HTTP 200 from $frontendUrl within 120 seconds."
    }

    [PSCustomObject]@{
        mode = if ($Production) { "production" } else { "development" }
        processes = @($startedProcesses)
        api_url = $apiBaseUrl
        frontend_url = $frontendUrl
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $processFile -Encoding utf8

    Write-Host "API health: HTTP $($healthResponse.StatusCode) $healthUrl"
    Write-Host "Frontend: HTTP 200 $frontendUrl"
    Write-Host "Mode: $(if ($Production) { 'production' } else { 'development' })"
    Write-Host "API PID: $($apiProcess.Id); frontend PID: $($frontendProcess.Id)"
    Start-Process $frontendUrl
}
catch {
    foreach ($entry in $startedProcesses) {
        Stop-LaunchedProcess -ProcessId $entry.pid
    }
    throw
}
