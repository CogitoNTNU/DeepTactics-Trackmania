@echo off
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
echo Press any key to start all components...
pause >nul

cd /d "%~dp0"

echo.
echo Starting Server...
start "TMRL Server" cmd /k "run_iqn_server.bat"
timeout /t 3 >nul

echo Starting Trainer...
start "TMRL Trainer (IQN)" cmd /k "run_iqn_trainer.bat"
timeout /t 3 >nul

echo Starting Worker...
start "TMRL Worker" cmd /k "run_iqn_worker.bat"

echo.
echo ====================================
echo All components started!
echo ====================================
echo.
echo You should now have 3 windows open:
echo   - Server (manages communication)
echo   - Trainer (trains the IQN model)
echo   - Worker (collects data from TrackMania)
echo.
echo To stop training, close all windows.
echo.
pause
