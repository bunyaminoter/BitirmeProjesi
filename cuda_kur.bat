@echo off
chcp 65001 >nul
color 0B
title CUDA Destekli PyTorch Kurulumu

echo =======================================================
echo     NVIDIA GPU (CUDA) DESTEKLI PYTORCH KURULUMU
echo =======================================================
echo.
echo Bilgisayarinizda RTX 5050 / NVIDIA ekran karti bulunuyor.
echo Modelin ekran kartinizi (GPU) kullanabilmesi icin PyTorch'un 
echo CPU versiyonu silinip, CUDA destekli versiyonu kurulacak.
echo.
pause

REM Sanal ortami aktif et
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Sanal ortam aktif edildi.
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Sanal ortam aktif edildi.
) else (
    echo UYARI: Sanal ortam bulunamadi, global python ortaminda kuruluyor...
)

echo.
echo [1/2] Mevcut (CPU) PyTorch siliniyor...
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo [2/2] CUDA (Ekran Karti) destekli PyTorch indiriliyor...
echo Bu islem internet hiziniza bagli olarak 10-15 dakika surebilir (Yaklasik 2.5 GB).
echo.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo =======================================================
echo Kurulum Tamamlandi!
echo Artik modeli egitirken otomatik olarak CUDA (Ekran karti) kullanilacak.
echo =======================================================
pause
