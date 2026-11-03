[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$processFile = Join-Path $repositoryRoot ".runtime\processes.json"

if (-not (Test-Path -LiteralPath $processFile -PathType Leaf)) {
    Write-Host "No PAPER2PRODUCT process record found; nothing to stop."
    exit 0
}

$record = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
foreach ($entry in $record.processes) {
    $process = Get-Process -Id ([int]$entry.pid) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Host "$($entry.name) PID $($entry.pid) is already stopped."
        continue
    }
    Stop-Process -Id ([int]$entry.pid) -Force
    Write-Host "Stopped $($entry.name) PID $($entry.pid)."
}
Remove-Item -LiteralPath $processFile -Force
Write-Host "PAPER2PRODUCT process record removed."
