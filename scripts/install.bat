@echo off
setlocal enabledelayedexpansion

echo Zerodha Algo Trader - Installation Script
echo =======================================

REM Set root directory and Python minimum version
set "ROOT_DIR=%~dp0.."
set "MIN_PYTHON_VER=3.11"

REM Verify Python installation and version
echo Checking Python installation...
python --version > temp.txt 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python %MIN_PYTHON_VER% or higher
    exit /b 1
)
set /p PYTHON_VER=<temp.txt
del temp.txt
echo Found: %PYTHON_VER%

REM Create fresh virtual environment
echo.
echo Creating virtual environment...
if exist "%ROOT_DIR%\venv" (
    echo Removing existing virtual environment...
    rmdir /s /q "%ROOT_DIR%\venv"
)
python -m venv "%ROOT_DIR%\venv"
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "%ROOT_DIR%\venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies with progress
echo.
echo Installing dependencies...
echo This may take a few minutes...
pip install -r "%ROOT_DIR%\requirements.txt"
if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

REM Success message and instructions
echo.
echo =======================================
echo Installation completed successfully!
echo.
echo To start using the trading bot:
echo 1. Activate the environment:
echo    %ROOT_DIR%\venv\Scripts\activate.bat
echo.
echo 2. Configure your settings:
echo    - Copy config/secrets.example.yaml to config/secrets.yaml
echo    - Edit config/secrets.yaml with your Zerodha credentials
echo.
echo 3. Start the bot:
echo    python main.py
echo =======================================

exit /b 0
