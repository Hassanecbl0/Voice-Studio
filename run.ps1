Set-Location $PSScriptRoot
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "              🎙️  XTTS Voice Studio" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Project Python was not found at: $python"
    exit 1
}

Write-Host "Starting Web UI (open the local URL printed by Gradio; normally http://127.0.0.1:7860)..." -ForegroundColor Green
& $python app.py
