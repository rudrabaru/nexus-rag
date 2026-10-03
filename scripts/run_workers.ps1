# Runs what is queued, then exits. Fetch first (it feeds the parse queue), then parse.
# Nothing polls the database while this is not running, so Neon's compute can sleep.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

foreach ($queue in @("fetch", "ingest")) {
    & $python -m src.jobs.workers $queue --drain
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
