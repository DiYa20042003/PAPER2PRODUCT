[CmdletBinding()]
param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendDirectory = Join-Path $repositoryRoot "backend"
$runsDirectory = Join-Path $backendDirectory "experiments\lenet_mnist\runs"
$databasePath = Join-Path $backendDirectory "data\paper2product.sqlite3"
$defaultOutputDirectory = Join-Path $repositoryRoot "local-backups"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = $defaultOutputDirectory
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot $OutputDirectory
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to include the committed source in the backup."
}
if (-not (Test-Path -LiteralPath $databasePath -PathType Leaf)) {
    throw "Local database is missing: $databasePath"
}
if (-not (Test-Path -LiteralPath $runsDirectory -PathType Container)) {
    throw "Experiment runs directory is missing: $runsDirectory"
}
$worktreeChanges = & git -C $repositoryRoot status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Git worktree."
}
if ($worktreeChanges) {
    throw "Commit or stash source changes before creating a source-complete backup."
}

$completedRuns = foreach ($metricsFile in Get-ChildItem -LiteralPath $runsDirectory -Filter "metrics.json" -File -Recurse) {
    try {
        $metrics = Get-Content -LiteralPath $metricsFile.FullName -Raw | ConvertFrom-Json
        $runDirectory = $metricsFile.Directory.FullName
        $configFile = Join-Path $runDirectory "config.json"
        $logFile = Join-Path $runDirectory "log.txt"
        if ($metrics.status -eq "completed" -and
            (Test-Path -LiteralPath $configFile -PathType Leaf) -and
            (Test-Path -LiteralPath $logFile -PathType Leaf)) {
            [PSCustomObject]@{
                Directory = $runDirectory
                Metrics = $metricsFile.FullName
                Config = $configFile
                Log = $logFile
                Timestamp = $metricsFile.LastWriteTimeUtc
                RunId = $metrics.run_id
            }
        }
    }
    catch {
        Write-Warning "Ignoring unreadable metrics file: $($metricsFile.FullName): $($_.Exception.Message)"
    }
}
$latestRun = $completedRuns | Sort-Object Timestamp -Descending | Select-Object -First 1
if ($null -eq $latestRun) {
    throw "No completed run with metrics.json, config.json, and log.txt was found."
}

$head = & git -C $repositoryRoot rev-parse --short HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Could not identify the source revision."
}
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archiveName = "paper2product-backup-$timestamp-$head.zip"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$archivePath = Join-Path $OutputDirectory $archiveName
$stagingDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString("N"))
$sourceArchive = Join-Path ([System.IO.Path]::GetTempPath()) ("paper2product-source-" + [guid]::NewGuid().ToString("N") + ".zip")

try {
    New-Item -ItemType Directory -Path $stagingDirectory | Out-Null
    & git -C $repositoryRoot archive --format=zip "--output=$sourceArchive" HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "git archive failed with exit code $LASTEXITCODE."
    }
    Expand-Archive -LiteralPath $sourceArchive -DestinationPath $stagingDirectory

    $runDestination = Join-Path $stagingDirectory "recorded-run"
    New-Item -ItemType Directory -Path $runDestination | Out-Null
    Copy-Item -LiteralPath $latestRun.Metrics -Destination (Join-Path $runDestination "metrics.json")
    Copy-Item -LiteralPath $latestRun.Config -Destination (Join-Path $runDestination "config.json")
    Copy-Item -LiteralPath $latestRun.Log -Destination (Join-Path $runDestination "log.txt")

    $databaseDestination = Join-Path $stagingDirectory "local-data"
    New-Item -ItemType Directory -Path $databaseDestination | Out-Null
    Copy-Item -LiteralPath $databasePath -Destination (Join-Path $databaseDestination "paper2product.sqlite3")

    Compress-Archive -Path (Join-Path $stagingDirectory "*") -DestinationPath $archivePath -CompressionLevel Optimal
    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        Write-Host "Backup archive: $archivePath"
        Write-Host "Source revision: $($head.Trim())"
        Write-Host "Recorded run: $($latestRun.RunId)"
        Write-Host "Archive contents:"
        foreach ($entry in $archive.Entries) {
            Write-Host ("  {0} ({1} bytes)" -f $entry.FullName, $entry.Length)
        }
    }
    finally {
        $archive.Dispose()
    }
}
finally {
    if (Test-Path -LiteralPath $stagingDirectory) {
        Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
    }
    if (Test-Path -LiteralPath $sourceArchive) {
        Remove-Item -LiteralPath $sourceArchive -Force
    }
}
