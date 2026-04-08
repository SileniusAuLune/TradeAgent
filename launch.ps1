#Requires -Version 5.1
<#
.SYNOPSIS
    TradeAgent — Launch Script
    Activates the virtual environment and starts the Streamlit dashboard.

.DESCRIPTION
    Double-click the desktop shortcut (created by install_windows.ps1)
    OR run directly:
        .\launch.ps1

    The app opens automatically in your default browser at http://localhost:8501
    Press Ctrl+C in this window to stop the server.
#>

$ErrorActionPreference = "Stop"
$REPO_DIR = $PSScriptRoot
$VENV_DIR = Join-Path $REPO_DIR ".venv"
$APP_FILE = Join-Path $REPO_DIR "app.py"

# ── Banner ─────────────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ______ " -ForegroundColor Cyan
Write-Host " |__  __| " -ForegroundColor Cyan
Write-Host "   | |_ __ __ _  __| | ___ " -ForegroundColor Cyan
Write-Host "   | | '__/ _' |/ _' |/ _ \  " -ForegroundColor Cyan
Write-Host "   | | | | (_| | (_| |  __/  Agent" -ForegroundColor Cyan
Write-Host "   |_|_|  \__,_|\__,_|\___| " -ForegroundColor Cyan
Write-Host ""
Write-Host "  AI-Powered Trading Analysis" -ForegroundColor White
Write-Host "  Powered by Claude claude-opus-4-6" -ForegroundColor Gray
Write-Host ""

# ── Sanity checks ──────────────────────────────────────────────────────────────
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "  Virtual environment not found." -ForegroundColor Red
    Write-Host "  Please run install_windows.ps1 first." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $APP_FILE)) {
    Write-Host "  app.py not found in $REPO_DIR" -ForegroundColor Red
    Write-Host "  Make sure you're running this from the TradeAgent folder." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Check .env and API key ─────────────────────────────────────────────────────
$ENV_FILE = Join-Path $REPO_DIR ".env"
if (-not (Test-Path $ENV_FILE)) {
    Write-Host "  WARNING: .env file not found." -ForegroundColor Yellow
    Write-Host "  Copy .env.example to .env and add your ANTHROPIC_API_KEY." -ForegroundColor Yellow
    Write-Host ""
} else {
    $envContent = Get-Content $ENV_FILE -Raw -ErrorAction SilentlyContinue
    if (-not ($envContent -match "ANTHROPIC_API_KEY=sk-ant-")) {
        Write-Host "  WARNING: ANTHROPIC_API_KEY does not look set in .env." -ForegroundColor Yellow
        Write-Host "  The app will show an error until the key is added." -ForegroundColor Yellow
        Write-Host ""
    }
}

# ── Launch Streamlit ───────────────────────────────────────────────────────────
$streamlitExe = Join-Path $VENV_DIR "Scripts\streamlit.exe"

if (-not (Test-Path $streamlitExe)) {
    Write-Host "  streamlit not found in virtual environment." -ForegroundColor Red
    Write-Host "  Re-run install_windows.ps1 to reinstall dependencies." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  Starting TradeAgent..." -ForegroundColor Green
Write-Host "  Browser will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host ""

Set-Location $REPO_DIR

# Run Streamlit — it opens the browser automatically
& $streamlitExe run $APP_FILE `
    --server.headless false `
    --browser.gatherUsageStats false `
    --theme.base dark `
    --theme.primaryColor "#00d4aa" `
    --theme.backgroundColor "#0e1117" `
    --theme.secondaryBackgroundColor "#1e1e2e" `
    --theme.textColor "#fafafa"

# ── If it exits ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  TradeAgent has stopped." -ForegroundColor Yellow
Read-Host "Press Enter to close"
