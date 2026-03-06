@echo off
setlocal enabledelayedexpansion
title BTC Arena AI — Live Prediction Engine
color 0A

:: ============================================================
::  BTC ARENA AI — Professional Launcher
::  Project  : arena-btc
::  Script   : src\btc_predictor_all_in_one.py
::  Author   : arena-btc team
:: ============================================================

cls

echo.
echo  [92m============================================================[0m
echo  [92m
echo     ██████╗ ████████╗ ██████╗     █████╗ ██████╗ ███████╗███╗   ██╗ █████╗
echo     ██╔══██╗╚══██╔══╝██╔════╝    ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗
echo     ██████╔╝   ██║   ██║         ███████║██████╔╝█████╗  ██╔██╗ ██║███████║
echo     ██╔══██╗   ██║   ██║         ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══██║
echo     ██████╔╝   ██║   ╚██████╗    ██║  ██║██║  ██║███████╗██║ ╚████║██║  ██║
echo     ╚═════╝    ╚═╝    ╚═════╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
echo  [0m
echo  [92m============================================================[0m
echo  [97m   AI-Powered Bitcoin Prediction Engine  ^|  arena-btc[0m
echo  [92m============================================================[0m
echo.

:: ── Step 1: Navigate to project root ────────────────────────
echo  [93m[1/4] Navigating to project directory...[0m
cd /d "C:\Users\satis\OneDrive\Desktop\arena-btc"
if errorlevel 1 (
    echo.
    echo  [91m[ERROR] Could not navigate to project directory.[0m
    echo  [91m        Expected: C:\Users\satis\OneDrive\Desktop\arena-btc[0m
    echo.
    pause
    exit /b 1
)
echo  [92m        OK — %CD%[0m
echo.

:: ── Step 2: Check virtual environment ───────────────────────
echo  [93m[2/4] Checking virtual environment...[0m
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo  [91m[ERROR] Virtual environment not found at .venv\[0m
    echo  [97m        To create it, run in this folder:[0m
    echo  [97m          python -m venv .venv[0m
    echo  [97m          .venv\Scripts\activate[0m
    echo  [97m          pip install -r requirements.txt[0m
    echo.
    pause
    exit /b 1
)

:: ── Step 3: Activate virtual environment ────────────────────
echo  [93m[3/4] Activating Python virtual environment...[0m
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo  [91m[ERROR] Failed to activate virtual environment.[0m
    echo.
    pause
    exit /b 1
)
echo  [92m        OK — .venv activated[0m
echo.

:: ── Step 4: Check trained models ────────────────────────────
echo  [93m[4/4] Checking trained models...[0m
if not exist "models\" (
    echo.
    echo  [91m[WARNING] models\ folder not found.[0m
    echo  [97m          Run TRAIN_MODELS.bat first to generate models.[0m
    echo.
    pause
    exit /b 1
)

set MODEL_COUNT=0
for %%f in (models\*.pkl models\*.h5 models\*.joblib models\*.json) do set /a MODEL_COUNT+=1
if %MODEL_COUNT%==0 (
    echo.
    echo  [91m[WARNING] No trained model files found in models\[0m
    echo  [97m          Run TRAIN_MODELS.bat first to train the models.[0m
    echo.
    pause
    exit /b 1
)
echo  [92m        OK — %MODEL_COUNT% model file(s) found[0m
echo.

:: ── Launch sequence ──────────────────────────────────────────
echo  [92m============================================================[0m
echo  [97m   Starting BTC Prediction Arena...[0m
echo  [92m============================================================[0m
echo.
echo  [97m   Data     : data\[0m
echo  [97m   Models   : models\[0m
echo  [97m   Logs     : logs\[0m
echo  [97m   Charts   : charts\[0m
echo.
echo  [94m   System running. Press Ctrl+C to stop safely.[0m
echo  [92m============================================================[0m
echo.

:: ── Run main AI system ───────────────────────────────────────
python src\btc_predictor_all_in_one.py

:: ── Post-exit handling ───────────────────────────────────────
echo.
echo  [92m============================================================[0m
if errorlevel 1 (
    echo  [91m   The prediction engine stopped with an error.[0m
    echo  [91m   Exit code: %errorlevel%[0m
    echo  [97m   Check logs\ for details.[0m
) else (
    echo  [93m   The prediction engine has stopped.[0m
    echo  [97m   Restart this launcher to run again.[0m
)
echo  [92m============================================================[0m
echo.
echo  [97m  Press any key to close this window...[0m
pause >nul
endlocal
