Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "          MediSense AI - Clinical Diagnostic Platform" -ForegroundColor White
Write-Host "    AI-Driven Disease Predictor & Prescription Generator" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Starting Python ML & Gemini Clinical Engine on port 8000..." -ForegroundColor Green
Start-Process -NoNewWindow powershell -ArgumentList "-Command", "wsl -d Ubuntu bash -c 'cd /mnt/c/Users/maram/Disease-predictor-AI-prescription-generator && /mnt/c/Users/maram/Disease-predictor-AI-prescription-generator/venv_linux/bin/python server.py'"

Write-Host "[2/2] Starting React Client on port 3000..." -ForegroundColor Green
Set-Location -Path "$PSScriptRoot\client"
npm run dev -- --host 0.0.0.0 --port 3000
