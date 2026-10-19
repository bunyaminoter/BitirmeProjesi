@echo off
chcp 65001 >nul
color 0A
title Hibrit TİD - Hızlı Başlatma Menüsü

:menu
cls
echo =======================================================
echo     TÜRK İŞARET DİLİ (TİD) TANIMA SİSTEMİ BAŞLATMA
echo =======================================================
echo.
echo Lutfen yapmak istediginiz islemi secin:
echo.
echo [1] Veri On Isleme (Preprocess) - MediaPipe 75 Nokta Cikarimi
echo [2] Standart Model Egitimi (Train)
echo [3] Modeli Validation'da Degerlendir (Evaluate)
echo [4] Modeli Test'te Degerlendir (son karar sonrasi)
echo [5] Tek Video ile Cikarim Yap (Inference)
echo [6] Yerel API Sunucusunu Baslat (Server - Port 8000)
echo [7] Cikis
echo.
set /p secim="Seciminiz (1-7): "

REM Sanal ortami otomatik aktif etme
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

if "%secim%"=="1" goto preprocess
if "%secim%"=="2" goto train_single
if "%secim%"=="3" goto evaluate_val
if "%secim%"=="4" goto evaluate_test
if "%secim%"=="5" goto inference
if "%secim%"=="6" goto server
if "%secim%"=="7" goto exit

echo.
echo Gecersiz secim. Lutfen tekrar deneyin.
pause
goto menu

:preprocess
cls
echo.
echo [ISLEM] TİD Veri On Isleme Baslatiliyor...
echo.
python scripts\preprocess_tid.py --config configs\experiment\autsl_10_mobilenet.yaml --data_dir "%AUTSL_DATA_DIR%" --splits train val test
echo.
pause
goto menu

:train_single
cls
echo.
echo [ISLEM] Standart Model Egitimi Baslatiliyor...
echo.
python scripts\train.py --config configs\experiment\autsl_10_mobilenet.yaml --cache_dir cache\autsl_10_features
echo.
pause
goto menu

:evaluate_val
cls
echo.
echo [ISLEM] Degerlendirme (Evaluate) Baslatiliyor...
echo.
python scripts\evaluate.py --config configs\experiment\autsl_10_mobilenet.yaml --checkpoint outputs\autsl_10_mobilenet\checkpoints\best_model.pt --split val --cache_dir cache\autsl_10_features
echo.
pause
goto menu

:evaluate_test
cls
echo.
echo [ISLEM] Kilitli test degerlendirmesi baslatiliyor...
echo.
python scripts\evaluate.py --config configs\experiment\autsl_10_mobilenet.yaml --checkpoint outputs\autsl_10_mobilenet\checkpoints\best_model.pt --split test --cache_dir cache\autsl_10_features
echo.
pause
goto menu

:inference
cls
echo.
echo [ISLEM] Cikarim (Inference) Baslatiliyor...
echo Lutfen test etmek istediginiz videonun yolunu yazin:
set /p video_path="Video Yolu: "
echo.
python scripts\inference.py --config configs\experiment\autsl_10_mobilenet.yaml --checkpoint outputs\autsl_10_mobilenet\checkpoints\best_model.pt --video "%video_path%"
echo.
pause
goto menu

:server
cls
echo.
echo [ISLEM] API ve Web Sunucusu Baslatiliyor...
echo.
python scripts\server.py --config configs\experiment\autsl_10_mobilenet.yaml --checkpoint outputs\autsl_10_mobilenet\checkpoints\best_model.pt --port 8000
echo.
pause
goto menu

:exit
exit
