@echo off
echo ====================================
echo Starting TMRL Trainer with IQN
echo ====================================
echo.
echo This will start the IQN training agent.
echo Make sure the server is running first!
echo.
echo Training metrics will be displayed here.
echo Keep this terminal open while training.
echo.

cd /d "%~dp0"
python tmrl/custom_actor_module.py --trainer

pause
