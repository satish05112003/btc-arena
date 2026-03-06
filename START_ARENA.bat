@echo off
title BTC Prediction Arena
color 0A

echo.
echo  ============================================
echo   BTC PREDICTION ARENA
echo   Training: Binance 5-year data
echo   Live    : Coinbase WebSocket
echo  ============================================
echo.

cd /d "%~dp0"

if not exist "venv\" (
    echo  First-time setup: creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt -q --disable-pip-version-check

if not exist "models\model_5m.pkl" (
    echo.
    echo  ============================================
    echo   No trained models found.
    echo   You need to train the models first.
    echo  ============================================
    echo.
    echo  Step 1: Make sure your dataset files exist:
    echo    data\btc_5m_5years.csv
    echo    data\btc_15m_5years.csv
    echo.
    echo  Step 2: Train the models (takes ~10-15 mins):
    echo    python src\train_model.py
    echo.
    echo  Step 3: Then run this launcher again.
    echo.
    pause
    exit
)

echo.
echo  Models found. Starting live prediction engine...
echo.
echo  ============================================
echo   Set Telegram alerts (optional):
echo   set TELEGRAM_BOT_TOKEN=your_token_here
echo   set TELEGRAM_CHAT_ID=your_chat_id_here
echo  ============================================
echo.
echo  Press Ctrl+C to stop.
echo.

python src\btc_predictor_all_in_one.py

pause
