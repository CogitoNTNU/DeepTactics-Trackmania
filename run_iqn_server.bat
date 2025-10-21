@echo off
echo ====================================
echo Starting TMRL Server with IQN
echo ====================================
echo.
echo This will start the central server for TMRL.
echo Keep this terminal open while training.
echo.

cd /d "%~dp0"
python tmrl/run_iqn.py --server

pause
