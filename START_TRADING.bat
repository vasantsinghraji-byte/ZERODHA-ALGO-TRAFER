@echo off
title AlgoTrader Pro - Starting...

echo.
echo  ========================================
echo    ALGOTRADER PRO v2.0
echo    Professional Trading Made Simple
echo  ========================================
echo.

REM Check if venv exists
if exist "venv\Scripts\python.exe" (
    echo Found virtual environment!
    echo Starting application...
    echo.
    venv\Scripts\python.exe run.py
) else (
    echo Virtual environment not found.
    echo Trying system Python...
    echo.
    python run.py
)

echo.
pause
