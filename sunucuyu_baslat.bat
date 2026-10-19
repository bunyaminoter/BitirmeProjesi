@echo off
chcp 65001 >nul
color 0A
title TİD Tanıma Sistemi - Sunucu Başlatıcı

echo =======================================================
echo     TİD TANIMA SİSTEMİ YEREL SUNUCUSU BAŞLATILIYOR
echo =======================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Sunucu baslatiliyor (Port 8000)...
echo Mobil uygulama ve web arayuzu baglantisi icin hazir.
echo.
python scripts\server.py --config configs\experiment\autsl_10_mobilenet.yaml --checkpoint outputs\autsl_10_mobilenet\checkpoints\best_model.pt --port 8000

pause
