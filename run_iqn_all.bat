@echo off
setlocal enabledelayedexpansion

echo ====================================
echo TMRL IQN Training - Start All
echo ====================================
echo.
echo This will open 3 terminals:
echo   1. Server (port 6666)
echo   2. Trainer (IQN training)
echo   3. Worker (data collection)
echo.
echo Make sure TrackMania 2020 is running before starting!
echo.

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

REM Verify Python version
echo Checking Python version...
python --version
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

echo Press any key to start all components...
pause >nul

echo.
echo Starting Server...
start "TMRL Server" cmd /k "cd /d "%PROJECT_ROOT%" && set "PYTHONPATH=%PROJECT_ROOT%" && .venv\Scripts\activate && python tmrl/run_iqn.py --server"
timeout /t 3 >nul

echo Starting Trainer...
start "TMRL Trainer (IQN)" cmd /k "cd /d "%PROJECT_ROOT%" && set "PYTHONPATH=%PROJECT_ROOT%" && .venv\Scripts\activate && python tmrl/run_iqn.py --trainer"
timeout /t 3 >nul

echo Starting Worker...
start "TMRL Worker" cmd /k "cd /d "%PROJECT_ROOT%" && set "PYTHONPATH=%PROJECT_ROOT%" && .venv\Scripts\activate && python tmrl/run_iqn.py --worker"

echo.
echo ====================================
echo All components started!
echo ====================================
echo.
pause
