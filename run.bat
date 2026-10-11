@echo off
echo ======================================================================
echo           MediSense AI - Clinical Diagnostic Platform
echo     AI-Driven Disease Predictor ^& Prescription Generator
echo ======================================================================

echo [1/2] Starting Python ML & Gemini Clinical Engine (Port 8000)...
start /B wsl -d Ubuntu bash -c "cd '/mnt/c/Users/maram/Disease-predictor-AI-prescription-generator' && /mnt/c/Users/maram/Disease-predictor-AI-prescription-generator/venv_linux/bin/python server.py"

echo [2/2] Starting React Client (Port 3000)...
cd client
npm run dev -- --host 0.0.0.0 --port 3000
