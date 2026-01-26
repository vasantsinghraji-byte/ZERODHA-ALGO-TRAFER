@echo off
REM ============================================================================
REM AlgoTrader Pro - Windows Startup Script
REM ============================================================================
REM
REM This is the SINGLE canonical entry point for Windows users.
REM All components are validated before application launch.
REM
REM Usage:
REM     start.bat                    - Start with GUI (default)
REM     start.bat --validate         - Validate components only
REM     start.bat --headless         - Start without GUI
REM     start.bat --legacy           - Use legacy startup
REM     start.bat --help             - Show help
REM
REM Exit Codes:
REM     0 - Success
REM     1 - Python not found
REM     2 - Required directory missing
REM     3 - Required file missing
REM     4 - Dependency check failed
REM     5 - Component validation failed
REM     6 - Application runtime error
REM
REM ============================================================================

setlocal EnableDelayedExpansion

REM === Configuration ===
set "APP_NAME=AlgoTrader Pro"
set "APP_VERSION=2.0"
set "SCRIPT_DIR=%~dp0"
set "PYTHON_MIN_VERSION=3.9"
set "LOG_DIR=%SCRIPT_DIR%data\logs"
set "CONFIG_DIR=%SCRIPT_DIR%config"

REM === Colors (Windows 10+) ===
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

REM === Parse Arguments ===
set "MODE=gui"
set "VALIDATE_ONLY=0"
set "SKIP_DEPS=0"

:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--validate" set "VALIDATE_ONLY=1" & shift & goto :parse_args
if /i "%~1"=="--headless" set "MODE=headless" & shift & goto :parse_args
if /i "%~1"=="--legacy" set "MODE=legacy" & shift & goto :parse_args
if /i "%~1"=="--skip-deps" set "SKIP_DEPS=1" & shift & goto :parse_args
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
shift
goto :parse_args
:end_parse

REM === Banner ===
call :print_banner

REM === Run Validations ===
call :validate_all
if errorlevel 1 (
    echo.
    echo %RED%[FAILED]%RESET% Startup validation failed. See errors above.
    echo.
    echo Run '%~nx0 --help' for usage information.
    exit /b %ERRORLEVEL%
)

REM === If validate-only mode, exit here ===
if "%VALIDATE_ONLY%"=="1" (
    echo.
    echo %GREEN%[PASSED]%RESET% All validations passed. System ready.
    exit /b 0
)

REM === Launch Application ===
call :launch_app
exit /b %ERRORLEVEL%


REM ============================================================================
REM FUNCTIONS
REM ============================================================================

:print_banner
echo.
echo %CYAN%========================================%RESET%
echo %CYAN%  %APP_NAME% v%APP_VERSION%%RESET%
echo %CYAN%  Professional Trading Made Simple%RESET%
echo %CYAN%========================================%RESET%
echo.
goto :eof


:show_help
call :print_banner
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --validate     Validate all components without starting the app
echo   --headless     Run without GUI (server/automated mode)
echo   --legacy       Use legacy startup (no component validation)
echo   --skip-deps    Skip Python dependency check
echo   --help, -h     Show this help message
echo.
echo Examples:
echo   %~nx0                    Start with GUI
echo   %~nx0 --validate         Check if all components are ready
echo   %~nx0 --headless         Run in headless mode
echo.
echo Exit Codes:
echo   0 - Success
echo   1 - Python not found or wrong version
echo   2 - Required directory missing
echo   3 - Required file missing
echo   4 - Dependency check failed
echo   5 - Component validation failed
echo   6 - Application runtime error
echo.
exit /b 0


:validate_all
echo [*] Starting validation...
echo.

REM Step 1: Check Python
call :check_python
if errorlevel 1 exit /b 1

REM Step 2: Check directory structure
call :check_directories
if errorlevel 1 exit /b 2

REM Step 3: Check required files
call :check_required_files
if errorlevel 1 exit /b 3

REM Step 4: Check dependencies (optional)
if "%SKIP_DEPS%"=="0" (
    call :check_dependencies
    if errorlevel 1 exit /b 4
)

REM Step 5: Validate components
call :check_components
if errorlevel 1 exit /b 5

echo.
echo %GREEN%[OK]%RESET% All validations passed!
goto :eof


:check_python
echo [1/5] Checking Python installation...

REM Try 'python' first
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
    goto :check_python_version
)

REM Try 'python3'
where python3 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python3"
    goto :check_python_version
)

REM Try 'py' (Python Launcher)
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py"
    goto :check_python_version
)

echo %RED%[ERROR]%RESET% Python not found!
echo.
echo Please install Python %PYTHON_MIN_VERSION% or higher from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
exit /b 1

:check_python_version
REM Get Python version
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do set "PY_VERSION=%%v"

REM Extract major.minor
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

REM Check minimum version (3.9+)
if %PY_MAJOR% lss 3 (
    echo %RED%[ERROR]%RESET% Python %PY_VERSION% is too old. Need %PYTHON_MIN_VERSION%+
    exit /b 1
)
if %PY_MAJOR% equ 3 if %PY_MINOR% lss 9 (
    echo %RED%[ERROR]%RESET% Python %PY_VERSION% is too old. Need %PYTHON_MIN_VERSION%+
    exit /b 1
)

echo       %GREEN%[OK]%RESET% Python %PY_VERSION% found (%PYTHON_CMD%)
goto :eof


:check_directories
echo [2/5] Checking directory structure...

set "MISSING_DIRS="
set "DIR_COUNT=0"

REM Required directories
call :check_dir "core"
call :check_dir "core\events"
call :check_dir "core\data"
call :check_dir "core\execution"
call :check_dir "core\infrastructure"
call :check_dir "core\risk"
call :check_dir "core\state"
call :check_dir "strategies"
call :check_dir "backtest"
call :check_dir "ml"
call :check_dir "ml\feature_store"
call :check_dir "indicators"
call :check_dir "ui"
call :check_dir "config"

if defined MISSING_DIRS (
    echo %RED%[ERROR]%RESET% Missing directories:
    echo %MISSING_DIRS%
    exit /b 2
)

echo       %GREEN%[OK]%RESET% All %DIR_COUNT% required directories present
goto :eof

:check_dir
set /a DIR_COUNT+=1
if not exist "%SCRIPT_DIR%%~1\" (
    set "MISSING_DIRS=!MISSING_DIRS!         - %~1\n"
)
goto :eof


:check_required_files
echo [3/5] Checking required files...

set "MISSING_FILES="
set "FILE_COUNT=0"

REM Core entry points
call :check_file "run.py" "Main entry point"
call :check_file "requirements.txt" "Dependencies manifest"

REM Core modules
call :check_file "core\__init__.py" "Core module"
call :check_file "core\bootstrap.py" "Bootstrap module"
call :check_file "core\events\__init__.py" "Events module"
call :check_file "core\trading_engine.py" "Trading engine"

REM Strategies
call :check_file "strategies\__init__.py" "Strategies module"
call :check_file "strategies\base.py" "Strategy base class"

REM Backtest
call :check_file "backtest\__init__.py" "Backtest module"
call :check_file "backtest\engine.py" "Backtest engine"

REM UI
call :check_file "ui\__init__.py" "UI module"
call :check_file "ui\app.py" "UI application"

REM Infrastructure
call :check_file "core\infrastructure\__init__.py" "Infrastructure module"
call :check_file "core\infrastructure\integration.py" "Infrastructure integration"

if defined MISSING_FILES (
    echo %RED%[ERROR]%RESET% Missing files:
    for %%f in (%MISSING_FILES%) do echo         - %%f
    exit /b 3
)

echo       %GREEN%[OK]%RESET% All %FILE_COUNT% required files present
goto :eof

:check_file
set /a FILE_COUNT+=1
if not exist "%SCRIPT_DIR%%~1" (
    set "MISSING_FILES=!MISSING_FILES! %~1"
)
goto :eof


:check_dependencies
echo [4/5] Checking Python dependencies...

REM Quick check for critical packages
%PYTHON_CMD% -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARN]%RESET% Dependencies not installed. Installing...
    echo.
    %PYTHON_CMD% -m pip install -r "%SCRIPT_DIR%requirements.txt" --quiet
    if errorlevel 1 (
        echo %RED%[ERROR]%RESET% Failed to install dependencies.
        echo.
        echo Try running manually:
        echo   %PYTHON_CMD% -m pip install -r requirements.txt
        exit /b 4
    )
    echo       %GREEN%[OK]%RESET% Dependencies installed successfully
) else (
    echo       %GREEN%[OK]%RESET% Core dependencies available
)
goto :eof


:check_components
echo [5/5] Validating Phase 8-12 components...

REM Run Python component validation
%PYTHON_CMD% -c "from core.bootstrap import validate_components_only; validate_components_only()" >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARN]%RESET% Component validation requires full check...
    echo.
    %PYTHON_CMD% "%SCRIPT_DIR%run.py" --validate
    if errorlevel 1 (
        echo %RED%[ERROR]%RESET% Component validation failed.
        exit /b 5
    )
) else (
    echo       %GREEN%[OK]%RESET% All components validated
)
goto :eof


:launch_app
echo.
echo [*] Launching %APP_NAME%...
echo.

REM Create logs directory if needed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Set environment variables
set "PYTHONPATH=%SCRIPT_DIR%"
set "ALGOTRADER_LOG_LEVEL=INFO"

REM Launch based on mode
if "%MODE%"=="legacy" (
    echo     Mode: Legacy (no component validation)
    %PYTHON_CMD% "%SCRIPT_DIR%run.py" --legacy
    exit /b %ERRORLEVEL%
)

if "%MODE%"=="headless" (
    echo     Mode: Headless (no GUI)
    %PYTHON_CMD% "%SCRIPT_DIR%run.py" --headless
    exit /b %ERRORLEVEL%
)

REM Default: GUI mode
echo     Mode: GUI
%PYTHON_CMD% "%SCRIPT_DIR%run.py"
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Application exited with code %EXIT_CODE%
    echo.
    echo Check logs at: %LOG_DIR%
    echo.
    pause
)

exit /b %EXIT_CODE%
