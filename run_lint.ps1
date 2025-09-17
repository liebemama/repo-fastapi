<#
Run Ruff lint + format in one go.

Usage:
  pwsh tools/run_lint.ps1
#>

param(
  [switch]$CheckOnly = $false
)

$ErrorActionPreference = "Stop"

function Ensure-Tool($name) {
  $exists = (Get-Command $name -ErrorAction SilentlyContinue) -ne $null
  if (-not $exists) {
    Write-Host "⚠️  '$name' not found in PATH. Installing into current venv..." -ForegroundColor Yellow
    pip install $name | Out-Host
  }
}

Ensure-Tool "ruff"

if ($CheckOnly) {
  Write-Host "🔎 Running: ruff check ." -ForegroundColor Cyan
  ruff check .
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "✅ Lint check passed." -ForegroundColor Green
  exit 0
}

Write-Host "🧹 Running: ruff check . --fix" -ForegroundColor Cyan
ruff check . --fix
if ($LASTEXITCODE -ne 0) {
  Write-Host "❌ Lint violations remain after auto-fix." -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "🖊️  Running: ruff format ." -ForegroundColor Cyan
ruff format .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "✅ Codebase is clean and formatted." -ForegroundColor Green
