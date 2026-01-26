<#
.SYNOPSIS
    AlgoTrader Pro - PowerShell Startup Script

.DESCRIPTION
    Unified entry point for AlgoTrader Pro on Windows.
    Validates all Phase 8-12 components before application launch.

.PARAMETER Validate
    Validate components only without starting the application.

.PARAMETER Headless
    Run in headless mode (no GUI).

.PARAMETER Legacy
    Use legacy startup (no component validation).

.PARAMETER Capital
    Initial trading capital (default: 100000).

.PARAMETER LogLevel
    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).

.PARAMETER SkipDeps
    Skip Python dependency check.

.EXAMPLE
    .\start.ps1
    Start with GUI (default mode).

.EXAMPLE
    .\start.ps1 -Validate
    Validate all components without starting.

.EXAMPLE
    .\start.ps1 -Headless -Capital 200000
    Run in headless mode with custom capital.

.NOTES
    Exit Codes:
        0 - Success
        1 - Python not found
        2 - Required directory missing
        3 - Required file missing
        4 - Dependency check failed
        5 - Component validation failed
        6 - Application runtime error
#>

[CmdletBinding()]
param(
    [switch]$Validate,
    [switch]$Headless,
    [switch]$Legacy,
    [switch]$SkipDeps,
    [int]$Capital = 100000,
    [ValidateSet('DEBUG', 'INFO', 'WARNING', 'ERROR')]
    [string]$LogLevel = 'INFO'
)

# === Configuration ===
$AppName = "AlgoTrader Pro"
$AppVersion = "2.0"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonMinVersion = [version]"3.9"
$LogDir = Join-Path $ScriptDir "data\logs"
$ConfigDir = Join-Path $ScriptDir "config"

# === Helper Functions ===

function Write-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  $AppName v$AppVersion" -ForegroundColor Cyan
    Write-Host "  Professional Trading Made Simple" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Step, [string]$Message)
    Write-Host "[$Step] $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "      [OK] $Message" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Message)
    Write-Host "      [FAILED] $Message" -ForegroundColor Red
}

function Write-Warn {
    param([string]$Message)
    Write-Host "      [WARN] $Message" -ForegroundColor Yellow
}

# === Validation Functions ===

function Test-Python {
    Write-Step "1/5" "Checking Python installation..."

    # Try to find Python
    $pythonCommands = @('python', 'python3', 'py')
    $pythonCmd = $null

    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = $cmd
                break
            }
        } catch {
            continue
        }
    }

    if (-not $pythonCmd) {
        Write-Fail "Python not found!"
        Write-Host ""
        Write-Host "Please install Python $PythonMinVersion or higher from:" -ForegroundColor Yellow
        Write-Host "  https://www.python.org/downloads/"
        Write-Host ""
        return $null
    }

    # Check version
    $versionOutput = & $pythonCmd --version 2>&1
    if ($versionOutput -match "Python (\d+\.\d+\.\d+)") {
        $installedVersion = [version]$Matches[1]

        if ($installedVersion -lt $PythonMinVersion) {
            Write-Fail "Python $installedVersion is too old. Need $PythonMinVersion+"
            return $null
        }

        Write-Success "Python $installedVersion found ($pythonCmd)"
        return $pythonCmd
    }

    Write-Fail "Could not determine Python version"
    return $null
}

function Test-Directories {
    Write-Step "2/5" "Checking directory structure..."

    $requiredDirs = @(
        "core",
        "core\events",
        "core\data",
        "core\execution",
        "core\infrastructure",
        "core\risk",
        "core\state",
        "strategies",
        "backtest",
        "ml",
        "ml\feature_store",
        "indicators",
        "ui",
        "config"
    )

    $missing = @()
    foreach ($dir in $requiredDirs) {
        $fullPath = Join-Path $ScriptDir $dir
        if (-not (Test-Path $fullPath -PathType Container)) {
            $missing += $dir
        }
    }

    if ($missing.Count -gt 0) {
        Write-Fail "Missing directories:"
        foreach ($dir in $missing) {
            Write-Host "         - $dir" -ForegroundColor Red
        }
        return $false
    }

    Write-Success "All $($requiredDirs.Count) required directories present"
    return $true
}

function Test-RequiredFiles {
    Write-Step "3/5" "Checking required files..."

    $requiredFiles = @(
        @{Path="run.py"; Desc="Main entry point"},
        @{Path="requirements.txt"; Desc="Dependencies manifest"},
        @{Path="core\__init__.py"; Desc="Core module"},
        @{Path="core\bootstrap.py"; Desc="Bootstrap module"},
        @{Path="core\events\__init__.py"; Desc="Events module"},
        @{Path="core\trading_engine.py"; Desc="Trading engine"},
        @{Path="strategies\__init__.py"; Desc="Strategies module"},
        @{Path="strategies\base.py"; Desc="Strategy base class"},
        @{Path="backtest\__init__.py"; Desc="Backtest module"},
        @{Path="backtest\engine.py"; Desc="Backtest engine"},
        @{Path="ui\__init__.py"; Desc="UI module"},
        @{Path="ui\app.py"; Desc="UI application"},
        @{Path="core\infrastructure\__init__.py"; Desc="Infrastructure module"},
        @{Path="core\infrastructure\integration.py"; Desc="Infrastructure integration"}
    )

    $missing = @()
    foreach ($file in $requiredFiles) {
        $fullPath = Join-Path $ScriptDir $file.Path
        if (-not (Test-Path $fullPath -PathType Leaf)) {
            $missing += $file.Path
        }
    }

    if ($missing.Count -gt 0) {
        Write-Fail "Missing files:"
        foreach ($file in $missing) {
            Write-Host "         - $file" -ForegroundColor Red
        }
        return $false
    }

    Write-Success "All $($requiredFiles.Count) required files present"
    return $true
}

function Test-Dependencies {
    param([string]$PythonCmd)

    Write-Step "4/5" "Checking Python dependencies..."

    # Quick check for pandas (critical dependency)
    $result = & $PythonCmd -c "import pandas" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Dependencies not installed. Installing..."
        Write-Host ""

        $reqFile = Join-Path $ScriptDir "requirements.txt"
        & $PythonCmd -m pip install -r $reqFile --quiet

        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to install dependencies."
            Write-Host ""
            Write-Host "Try running manually:" -ForegroundColor Yellow
            Write-Host "  $PythonCmd -m pip install -r requirements.txt"
            return $false
        }

        Write-Success "Dependencies installed successfully"
    } else {
        Write-Success "Core dependencies available"
    }

    return $true
}

function Test-Components {
    param([string]$PythonCmd)

    Write-Step "5/5" "Validating Phase 8-12 components..."

    $runScript = Join-Path $ScriptDir "run.py"
    $result = & $PythonCmd $runScript --validate 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Component validation failed."
        Write-Host ""
        Write-Host $result -ForegroundColor Red
        return $false
    }

    Write-Success "All components validated"
    return $true
}

# === Main Script ===

Write-Banner

Write-Host "[*] Starting validation..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
$pythonCmd = Test-Python
if (-not $pythonCmd) {
    exit 1
}

# Step 2: Check directories
if (-not (Test-Directories)) {
    exit 2
}

# Step 3: Check required files
if (-not (Test-RequiredFiles)) {
    exit 3
}

# Step 4: Check dependencies
if (-not $SkipDeps) {
    if (-not (Test-Dependencies -PythonCmd $pythonCmd)) {
        exit 4
    }
}

# Step 5: Validate components
if (-not (Test-Components -PythonCmd $pythonCmd)) {
    exit 5
}

Write-Host ""
Write-Host "[OK] All validations passed!" -ForegroundColor Green

# If validate-only mode, exit here
if ($Validate) {
    Write-Host ""
    Write-Host "System ready for launch." -ForegroundColor Cyan
    exit 0
}

# Launch application
Write-Host ""
Write-Host "[*] Launching $AppName..." -ForegroundColor Cyan
Write-Host ""

# Create logs directory
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Set environment
$env:PYTHONPATH = $ScriptDir
$env:ALGOTRADER_LOG_LEVEL = $LogLevel

$runScript = Join-Path $ScriptDir "run.py"
$arguments = @()

if ($Legacy) {
    Write-Host "    Mode: Legacy (no component validation)" -ForegroundColor Yellow
    $arguments += "--legacy"
} elseif ($Headless) {
    Write-Host "    Mode: Headless (no GUI)" -ForegroundColor Cyan
    $arguments += "--headless"
} else {
    Write-Host "    Mode: GUI" -ForegroundColor Cyan
}

$arguments += "--capital", $Capital
$arguments += "--log-level", $LogLevel

# Launch
& $pythonCmd $runScript @arguments
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Application exited with code $exitCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check logs at: $LogDir" -ForegroundColor Yellow
}

exit $exitCode
