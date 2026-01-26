@echo off
REM ============================================================================
REM AlgoTrader Pro - Quick Launch (Double-click to start!)
REM ============================================================================
REM
REM This is a simple launcher for everyday use.
REM Double-click this file to start the trading application.
REM
REM For advanced options, use: start.bat --help
REM ============================================================================

title AlgoTrader Pro

cd /d "%~dp0"

REM Check if Python exists
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python is not installed!
    echo.
    echo  Please install Python 3.9+ from:
    echo    https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Launch with full validation
call start.bat

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Press any key to close...
    pause >nul
)
