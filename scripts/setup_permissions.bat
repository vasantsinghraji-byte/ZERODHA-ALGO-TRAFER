@echo off
setlocal enabledelayedexpansion

REM Enable error logging
set "LOG_FILE=%TEMP%\zerodha_setup.log"
echo Setup started at %DATE% %TIME% > "%LOG_FILE%"

REM Set root directory path
set "ROOT_DIR=%~dp0.."
cd /d "%ROOT_DIR%" 2>>"%LOG_FILE%"

echo Setting up permissions for: %ROOT_DIR%

REM Check if root directory exists with better error handling
if not exist "%ROOT_DIR%" (
    echo Error: Project directory not found: %ROOT_DIR%
    echo Details in: %LOG_FILE%
    exit /b 1
)

REM Try each operation separately with error handling
echo Step 1/4: Setting base permissions...
icacls "%ROOT_DIR%" /grant "%USERNAME%":(OI)(CI)F /T /Q >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Warning: Base permission setup partial - continuing...
)

echo Step 2/4: Creating directories...
if not exist "%ROOT_DIR%\logs" (
    mkdir "%ROOT_DIR%\logs" 2>>"%LOG_FILE%"
)

if not exist "%ROOT_DIR%\venv" (
    mkdir "%ROOT_DIR%\venv" 2>>"%LOG_FILE%"
)

echo Step 3/4: Setting logs permissions...
icacls "%ROOT_DIR%\logs" /grant "%USERNAME%":(OI)(CI)F /Q >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Retrying logs permissions with reduced scope...
    icacls "%ROOT_DIR%\logs" /grant "%USERNAME%":F /Q >>"%LOG_FILE%" 2>&1
)

echo Step 4/4: Setting venv permissions...
icacls "%ROOT_DIR%\venv" /grant "%USERNAME%":(OI)(CI)F /Q >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Retrying venv permissions with reduced scope...
    icacls "%ROOT_DIR%\venv" /grant "%USERNAME%":F /Q >>"%LOG_FILE%" 2>&1
)

REM Verify critical paths
set "SETUP_OK=1"
if not exist "%ROOT_DIR%\logs" set "SETUP_OK=0"
if not exist "%ROOT_DIR%\venv" set "SETUP_OK=0"

if "%SETUP_OK%"=="0" (
    echo Error: Critical directories missing. See %LOG_FILE% for details
    exit /b 1
)

echo Setup completed - Log file: %LOG_FILE%
exit /b 0
