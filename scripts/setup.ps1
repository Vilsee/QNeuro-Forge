# Run from the repository root:   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path ".venv")) { python -m venv .venv }
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[dev,study,demo]"
& $py -m playwright install chromium

Write-Host "`n--- tests ---" -ForegroundColor Green
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

Write-Host "`n--- rebuild and sign the study ---" -ForegroundColor Green
& .\.venv\Scripts\qneuro.exe study build
& .\.venv\Scripts\qneuro.exe study verify
if ($LASTEXITCODE -ne 0) { throw "verification failed" }

Write-Host "`nReady. Next:  .\scripts\demo.ps1" -ForegroundColor Green
