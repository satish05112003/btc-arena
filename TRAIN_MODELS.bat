@echo off
title BTC Prediction Arena - Model Training
color 0B

echo.
echo  ============================================
echo   BTC PREDICTION ARENA — Model Training
echo   Data source: Binance 5-year historical
echo  ============================================
echo.

cd /d "%~dp0"

if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt -q --disable-pip-version-check

echo.
echo  Checking for dataset files...
echo.

if not exist "data\btc_5m_5years.csv" (
    echo  ERROR: btc_5m_5years.csv not found at data\
    echo  Please run the downloader first.
    pause
    exit
)

if not exist "data\btc_15m_5years.csv" (
    echo  ERROR: btc_15m_5years.csv not found at data\
    echo  Please run the downloader first.
    pause
    exit
)

echo  Datasets found. Starting training...
echo  This will take approximately 10-20 minutes.
echo.

python src\train_model.py

echo.
echo  ============================================
echo   Training complete!
echo   Models saved to: models\
echo   Run START_ARENA.bat to begin live trading.
echo  ============================================
echo.
pause
