@echo off
:: Set terminal color (0A = black background, bright green text)
color 0A

echo ====================================================
echo                STREAMLIT APP LAUNCHER
echo ====================================================
echo.

echo [1/3] Checking Python installation...
python --version || (
    echo Python not found.
    pause
    exit /b
)

echo [2/3] Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Found error while installing dependencies
    pause
    exit /b
)

echo [3/3] Starting Streamlit app...
echo ----------------------------------------------------
streamlit run app.py

echo ----------------------------------------------------
echo App closed
pause