@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Oyun Alani Izleme uygulamasi baslatiliyor...
python desktop_app.py
if errorlevel 1 (
    echo.
    echo Uygulama bir hatayla kapandi. Yukaridaki mesaji kontrol et.
    pause
)
