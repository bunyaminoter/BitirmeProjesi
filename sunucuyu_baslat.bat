@echo off
chcp 65001 >nul
color 0A
title ASL Tanima Sistemi - Sunucu Baslatici

echo =======================================================
echo     ASL TANIMA SISTEMI YEREL SUNUCUSU BASLATILIYOR
echo =======================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Sunucu baslatiliyor (Port 8000)...
echo Mobil uygulama baglantisi icin hazir.
echo.
python scripts\server.py --config configs\experiment\asl_citizen_baseline.yaml --checkpoint outputs\asl_citizen_baseline\checkpoints\best_model.pt --port 8000

pause
