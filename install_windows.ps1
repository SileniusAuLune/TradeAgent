#Requires -Version 5.1
<#
.SYNOPSIS
    TradeAgent - Windows Installer
    Sets up Python dependencies and creates a desktop shortcut.

.DESCRIPTION
    Run this once after cloning the repo:
        Right-click install_windows.ps1 -> "Run with PowerShell"
    Or from a PowerShell window:
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
        .\install_windows.ps1

.NOTES
    Requirements:
        - Python 3.10 or later  (https://www.python.org/downloads/)
        - An Anthropic API key  (https://console.anthropic.com)
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REPO_DIR = $PSScriptRoot
$ENV_FILE  = Join-Path $REPO_DIR ".env"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TradeAgent - Windows Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                $pythonCmd = $cmd
                Write-Host "  Found: $ver" -ForegroundColor Green
                break
            }
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host ""
    Write-Host "  ERROR: Python 3.10+ is required but was not found." -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  Make sure to check 'Add Python to PATH' during install." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# 2. Create virtual environment
Write-Host "[2/5] Setting up virtual environment..." -ForegroundColor Yellow

$venvDir = Join-Path $REPO_DIR ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "  Creating virtual environment at .venv ..." -ForegroundColor Gray
    & $pythonCmd -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "  Created." -ForegroundColor Green
} else {
    Write-Host "  Virtual environment already exists - skipping." -ForegroundColor Green
}

$pipExe = Join-Path $venvDir "Scripts\pip.exe"
$pyExe  = Join-Path $venvDir "Scripts\python.exe"

# 3. Install dependencies
Write-Host "[3/5] Installing Python packages (this may take a minute)..." -ForegroundColor Yellow

$reqFile = Join-Path $REPO_DIR "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Host "  ERROR: requirements.txt not found in $REPO_DIR" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

& $pipExe install --upgrade pip --quiet
& $pipExe install -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: pip install failed. Check your internet connection." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  All packages installed." -ForegroundColor Green

# 4. .env file setup
Write-Host "[4/5] Checking .env configuration..." -ForegroundColor Yellow

if (-not (Test-Path $ENV_FILE)) {
    $exampleEnv = Join-Path $REPO_DIR ".env.example"
    if (Test-Path $exampleEnv) {
        Copy-Item $exampleEnv $ENV_FILE
        Write-Host "  Created .env from .env.example" -ForegroundColor Green
    } else {
        Set-Content $ENV_FILE "ANTHROPIC_API_KEY=your_api_key_here"
        Write-Host "  Created blank .env" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "  ACTION REQUIRED: Open .env and add your Anthropic API key." -ForegroundColor Yellow
    Write-Host "  File location: $ENV_FILE" -ForegroundColor Yellow
    Write-Host ""
} else {
    $envContent = Get-Content $ENV_FILE -Raw
    if ($envContent -match "ANTHROPIC_API_KEY=sk-ant-") {
        Write-Host "  API key already configured." -ForegroundColor Green
    } else {
        Write-Host "  .env exists but ANTHROPIC_API_KEY may not be set." -ForegroundColor Yellow
        Write-Host "  Edit: $ENV_FILE" -ForegroundColor Yellow
    }
}

# 5. Create desktop shortcut
Write-Host "[5/5] Creating desktop shortcut..." -ForegroundColor Yellow

$launchScript = Join-Path $REPO_DIR "launch.ps1"
$desktopPath  = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "TradeAgent.lnk"

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)

$shortcut.TargetPath       = "powershell.exe"
$shortcut.Arguments        = "-ExecutionPolicy Bypass -WindowStyle Normal -File `"$launchScript`""
$shortcut.WorkingDirectory = $REPO_DIR
$shortcut.WindowStyle      = 1
$shortcut.Description      = "Launch TradeAgent - AI Trading Analysis"

$pythonIconPath = Join-Path $venvDir "Scripts\python.exe"
if (Test-Path $pythonIconPath) {
    $shortcut.IconLocation = "$pythonIconPath,0"
}

$shortcut.Save()

if (Test-Path $shortcutPath) {
    Write-Host "  Desktop shortcut created: TradeAgent.lnk" -ForegroundColor Green
} else {
    Write-Host "  Could not create desktop shortcut (permissions?)." -ForegroundColor Yellow
    Write-Host "  You can still launch via: .\launch.ps1" -ForegroundColor Yellow
}

# Done
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Edit .env and set ANTHROPIC_API_KEY=sk-ant-..." -ForegroundColor White
Write-Host "  2. Double-click 'TradeAgent' on your Desktop" -ForegroundColor White
Write-Host "     OR run: .\launch.ps1" -ForegroundColor White
Write-Host "  3. Your browser will open to http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Optional (Schwab live trading):" -ForegroundColor White
Write-Host "  - Add SCHWAB_API_KEY and SCHWAB_API_SECRET to .env" -ForegroundColor White
Write-Host "  - See QUICKSTART.md for full setup guide" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
