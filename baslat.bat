@echo off
chcp 65001 >nul
color 0A
title Hibrit ASL - Hızlı Başlatma Menüsü

:menu
cls
echo =======================================================
echo     HIBRIT ASL TANIMA SISTEMI HIZLI BASLATMA MENUSU
echo =======================================================
echo.
echo Lutfen yapmak istediginiz islemi secin:
echo.
echo [1] Veri On Isleme (Preprocess) - MediaPipe Oznitelik Cikarimi
echo [2] Modeli Egit (Train)
echo [3] Modeli Test Et (Evaluate)
echo [4] Tek Video ile Cikarim Yap (Inference)
echo [5] Cikis
echo.
set /p secim="Seciminiz (1-5): "

REM Sanal ortami otomatik aktif etme
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

if "%secim%"=="1" goto preprocess
if "%secim%"=="2" goto train
if "%secim%"=="3" goto evaluate
if "%secim%"=="4" goto inference
if "%secim%"=="5" goto exit

echo.
echo Gecersiz secim. Lutfen tekrar deneyin.
pause
goto menu

:preprocess
cls
echo.
echo [ISLEM] Veri On Isleme (Preprocess) Baslatiliyor...
echo.
python scripts\preprocess.py --config configs\experiment\wlasl100_baseline.yaml --cache_dir cache\features\
echo.
pause
goto menu

:train
cls
echo.
echo [ISLEM] Egitim (Train) Baslatiliyor...
echo.
python scripts\train.py --config configs\experiment\wlasl100_baseline.yaml
echo.
pause
goto menu

:evaluate
cls
echo.
echo [ISLEM] Degerlendirme (Evaluate) Baslatiliyor...
echo (Varsayilan olarak best_model.pt kullanilir)
echo.
python scripts\evaluate.py --config configs\experiment\wlasl100_baseline.yaml --checkpoint outputs\wlasl100_baseline\checkpoints\best_model.pt
echo.
pause
goto menu

:inference
cls
echo.
echo [ISLEM] Cikarim (Inference) Baslatiliyor...
echo Lutfen test etmek istediginiz videonun yolunu yazin (orn: videos\05237.mp4)
set /p video_path="Video Yolu: "
echo.
python scripts\inference.py --config configs\experiment\wlasl100_baseline.yaml --checkpoint outputs\wlasl100_baseline\checkpoints\best_model.pt --video "%video_path%"
echo.
pause
goto menu

:exit
exit
