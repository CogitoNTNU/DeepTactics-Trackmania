@echo off
echo ====================================
echo Starting TMRL Worker with IQN
echo ====================================
echo.
echo This will start a rollout worker to collect data.
echo Make sure:
echo   1. The server is running
echo   2. TrackMania 2020 is open
echo.
echo The worker will use your IQN actor to play.
echo Keep this terminal open while training.
echo.

cd /d "%~dp0"
python tmrl/run_iqn.py --worker

pause
