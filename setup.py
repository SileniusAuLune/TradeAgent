#!/usr/bin/env python3
"""
TradeAgent setup script.

Run once on a new machine (or after pulling to a new desktop):
    python setup.py

Handles:
  - ANTHROPIC_API_KEY verification
  - Itradedash DB auto-detection across common Windows/Mac/Linux paths
  - Schwab API credentials (optional)
  - .env file creation / update (existing values are never overwritten)
  - SQLite DB initialisation and JSON migration
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# ── Helpers ────────────────────────────────────────────────────────────────────

ENV_FILE = Path(".env")
PYTHON   = sys.executable


def _read_env() -> dict[str, str]:
    """Parse existing .env into a dict (comments/blanks ignored)."""
    if not ENV_FILE.exists():
        return {}
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def _write_env(env: dict[str, str]) -> None:
    """
    Write env dict back to .env, preserving all existing comments and structure.
    Only adds keys that are missing; never overwrites existing values.
    """
    existing_text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    lines         = existing_text.splitlines()

    # Keys already present (with any value, even placeholder)
    present = set()
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            present.add(line.split("=", 1)[0].strip())

    # Append only genuinely new keys
    additions = []
    for k, v in env.items():
        if k not in present and v:
            additions.append(f"{k}={v}")

    if additions:
        sep = "\n" if existing_text.endswith("\n") else "\n\n"
        new_text = existing_text + sep + "\n".join(additions) + "\n"
        ENV_FILE.write_text(new_text, encoding="utf-8")


def _ask(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        val = ""
    return val if val else default


def _ok(msg: str)  -> None: print(f"  [OK] {msg}")
def _warn(msg: str)-> None: print(f"  [!!] {msg}")
def _info(msg: str)-> None: print(f"  --> {msg}")

SEP = "-" * 60


# ── Step 1: Anthropic API key ──────────────────────────────────────────────────

def check_anthropic(env: dict) -> dict:
    print(f"\n{SEP}")
    print("1. Anthropic API Key")
    print(SEP)

    key = env.get("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")

    if key and not key.startswith("your_"):
        _ok(f"Key found: sk-ant-...{key[-6:]}")
        return {}

    print("  Get your key at https://console.anthropic.com")
    key = _ask("  Paste ANTHROPIC_API_KEY: ")
    if key and key.startswith("sk-ant-"):
        _ok("Key accepted.")
        return {"ANTHROPIC_API_KEY": key}
    else:
        _warn("No valid key entered — the agent will not work without it.")
        return {}


# ── Step 2: Itradedash DB ──────────────────────────────────────────────────────

# Ordered list of paths to probe (most likely first).
# {user} is replaced with the current user's home directory.
_ITRADEDASH_CANDIDATES = [
    # Same parent directory as TradeAgent (typical sibling-repo layout)
    Path(__file__).parent.parent / "Itradedash" / "data" / "insider_trades.db",
    # Windows Documents folder
    Path.home() / "Documents" / "Itradedash" / "data" / "insider_trades.db",
    # Home root
    Path.home() / "Itradedash" / "data" / "insider_trades.db",
    # Desktop
    Path.home() / "Desktop" / "Itradedash" / "data" / "insider_trades.db",
    # Drive root (Windows common clone location)
    Path("C:/Itradedash/data/insider_trades.db"),
    Path("C:/Users") / os.getenv("USERNAME", "") / "Itradedash" / "data" / "insider_trades.db",
]


def check_itradedash(env: dict) -> dict:
    print(f"\n{SEP}")
    print("2. Itradedash Insider Signal DB")
    print(SEP)
    print("  Repo: https://github.com/SileniusAuLune/Itradedash")

    existing_db  = env.get("ITRADEDASH_DB", "").strip()
    existing_api = env.get("ITRADEDASH_API", "").strip()

    updates: dict[str, str] = {}

    # Already configured
    if existing_db and Path(existing_db).exists():
        _ok(f"DB already configured and found: {existing_db}")
        return {}
    if existing_db and not Path(existing_db).exists():
        _warn(f"ITRADEDASH_DB is set but file not found: {existing_db}")

    # Auto-detect
    found: Optional[Path] = None
    for candidate in _ITRADEDASH_CANDIDATES:
        if candidate.exists():
            found = candidate.resolve()
            break

    if found:
        _ok(f"Auto-detected: {found}")
        confirm = _ask("  Use this path? [Y/n]: ", "y").lower()
        if confirm != "n":
            updates["ITRADEDASH_DB"] = str(found).replace("\\", "/")
            return updates

    # Not found — offer options
    _info("Itradedash DB not found automatically.")
    print("  Options:")
    print("  [1] Enter the path manually")
    print("  [2] Use HTTP API instead (requires Itradedash Flask server running)")
    print("  [3] Skip — configure later by setting ITRADEDASH_DB in .env")
    choice = _ask("  Choice [1/2/3]: ", "3")

    if choice == "1":
        path = _ask("  Full path to insider_trades.db: ").strip().replace("\\", "/")
        if path and Path(path).exists():
            _ok(f"Verified: {path}")
            updates["ITRADEDASH_DB"] = path
        elif path:
            _warn("Path entered but file not found — saving anyway, fix when Itradedash is set up.")
            updates["ITRADEDASH_DB"] = path
        else:
            _warn("No path entered — skipping.")

    elif choice == "2":
        api = _ask("  Itradedash API URL [http://localhost:8080]: ", "http://localhost:8080")
        updates["ITRADEDASH_API"] = api
        _info("Will fall back to API — make sure Itradedash server is running before trading.")

    else:
        _info("Skipped. Set ITRADEDASH_DB in .env when ready.")
        if not existing_api:
            updates["ITRADEDASH_API"] = "http://localhost:8080"

    return updates


# ── Step 3: Schwab (optional) ──────────────────────────────────────────────────

def check_schwab(env: dict) -> dict:
    print(f"\n{SEP}")
    print("3. Charles Schwab API (optional — for live trading)")
    print(SEP)

    key    = env.get("SCHWAB_API_KEY", "")
    secret = env.get("SCHWAB_API_SECRET", "")

    if key and not key.startswith("your_") and secret and not secret.startswith("your_"):
        _ok("Schwab credentials already configured.")
        return {}

    _info("Skip this to run in paper-trading mode only.")
    configure = _ask("  Configure Schwab now? [y/N]: ", "n").lower()
    if configure != "y":
        _info("Skipped — paper trading mode will be used.")
        return {}

    print("  Register at https://developer.schwab.com")
    print("  Callback URL must be: https://127.0.0.1")
    key    = _ask("  SCHWAB_API_KEY: ")
    secret = _ask("  SCHWAB_API_SECRET: ")
    updates = {}
    if key:    updates["SCHWAB_API_KEY"]    = key
    if secret: updates["SCHWAB_API_SECRET"] = secret
    if updates:
        _ok("Schwab credentials saved.")
    return updates


# ── Step 4: DB init + migration ────────────────────────────────────────────────

def init_database() -> None:
    print(f"\n{SEP}")
    print("4. Database Setup")
    print(SEP)

    db_path = Path("trades.db")

    if not db_path.exists():
        _info("Initialising trades.db...")
        result = subprocess.run([PYTHON, "db.py", "--migrate"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                _info(line)
            _ok("Database ready.")
        else:
            _warn(f"DB init failed: {result.stderr.strip()}")
    else:
        _ok(f"trades.db already exists ({db_path.stat().st_size // 1024} KB)")
        result = subprocess.run([PYTHON, "db.py", "--stats"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                _info(line)


# ── Step 5: Import bundle if present ──────────────────────────────────────────

def import_bundle() -> None:
    bundles = sorted(Path(".").glob("tradeagent_export_*.json"), reverse=True)
    if not bundles:
        return

    latest = bundles[0]
    print(f"\n{SEP}")
    print("5. Restore from Export Bundle")
    print(SEP)
    _info(f"Found: {latest.name}")

    confirm = _ask("  Import this bundle? (restores trade history + live state) [Y/n]: ", "y").lower()
    if confirm == "n":
        _info("Skipped.")
        return

    result = subprocess.run(
        [PYTHON, "db.py", "--import", str(latest)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            _info(line)
        _ok("Bundle imported.")
    else:
        _warn(f"Import failed: {result.stderr.strip()}")


# ── Step 6: Verify insider connection ─────────────────────────────────────────

def verify_insider() -> None:
    # Reload env so the just-written values are picked up
    from dotenv import load_dotenv
    load_dotenv(override=True)

    result = subprocess.run(
        [PYTHON, "insider_intel.py"],
        capture_output=True, text=True,
    )
    status = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"
    if "connected" in status.lower():
        _ok(status)
    else:
        _warn(status)


# ── Step 7: Desktop shortcut ───────────────────────────────────────────────────

def create_desktop_shortcut() -> None:
    print(f"\n{SEP}")
    print("7. Desktop Shortcut")
    print(SEP)

    project_dir = Path(__file__).parent.resolve()
    desktop     = Path.home() / "Desktop"

    if sys.platform == "win32":
        _create_shortcut_windows(project_dir, desktop)
    elif sys.platform == "darwin":
        _create_shortcut_mac(project_dir, desktop)
    else:
        _create_shortcut_linux(project_dir, desktop)


def _get_windows_desktop() -> Path:
    """Return the real Desktop path, handling OneDrive redirection correctly."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, timeout=10,
        )
        path = result.stdout.strip()
        if path and Path(path).exists():
            return Path(path)
    except Exception:
        pass
    # Fallback: try both OneDrive and standard Desktop
    for candidate in [
        Path.home() / "OneDrive" / "Desktop",
        Path.home() / "Desktop",
    ]:
        if candidate.exists():
            return candidate
    # Last resort: return standard path even if it doesn't exist
    return Path.home() / "Desktop"


def _create_shortcut_windows(project_dir: Path, desktop: Path) -> None:
    # Override desktop with OneDrive-aware path
    desktop = _get_windows_desktop()

    launch_ps1 = project_dir / "launch.ps1"
    lnk_path   = desktop / "TradeAgent.lnk"

    if lnk_path.exists():
        _ok(f"Shortcut already exists: {lnk_path}")
        return

    try:
        ps = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$lnk = $ws.CreateShortcut("{lnk_path}"); '
            f'$lnk.TargetPath = "powershell.exe"; '
            f'$lnk.Arguments = \'-ExecutionPolicy Bypass -WindowStyle Normal -File "{launch_ps1}"\'; '
            f'$lnk.WorkingDirectory = "{project_dir}"; '
            f'$lnk.Description = "TradeAgent - AI Trading Dashboard"; '
            f'$lnk.Save()'
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and lnk_path.exists():
            _ok(f"Shortcut created: {lnk_path}")
        else:
            _warn(f"PowerShell shortcut failed: {result.stderr.strip()}")
            _create_shortcut_windows_bat(project_dir, desktop)
    except Exception as e:
        _warn(f"Shortcut error: {e}")
        _create_shortcut_windows_bat(project_dir, desktop)


def _create_shortcut_windows_bat(project_dir: Path, desktop: Path) -> None:
    """Fallback: write a .bat launcher to the desktop."""
    bat_path = desktop / "TradeAgent.bat"
    if bat_path.exists():
        _ok(f"Launcher already exists: {bat_path}")
        return
    bat_path.write_text(
        f'@echo off\n'
        f'cd /d "{project_dir}"\n'
        f'"{project_dir / ".venv" / "Scripts" / "streamlit.exe"}" run app.py\n'
        f'pause\n',
        encoding="utf-8",
    )
    _ok(f"Launcher created: {bat_path}")


def _create_shortcut_mac(project_dir: Path, desktop: Path) -> None:
    app_path = desktop / "TradeAgent.command"
    if app_path.exists():
        _ok(f"Launcher already exists: {app_path}")
        return
    app_path.write_text(
        f"#!/bin/bash\n"
        f'cd "{project_dir}"\n'
        f'source .venv/bin/activate\n'
        f"streamlit run app.py\n",
        encoding="utf-8",
    )
    app_path.chmod(0o755)
    _ok(f"Launcher created: {app_path}  (double-click to open)")


def _create_shortcut_linux(project_dir: Path, desktop: Path) -> None:
    # Try XDG .desktop file first; fall back to shell script
    desktop_file = desktop / "TradeAgent.desktop"
    if desktop_file.exists():
        _ok(f"Launcher already exists: {desktop_file}")
        return
    streamlit = project_dir / ".venv" / "bin" / "streamlit"
    desktop_file.write_text(
        f"[Desktop Entry]\n"
        f"Version=1.0\n"
        f"Type=Application\n"
        f"Name=TradeAgent\n"
        f"Comment=AI Trading Dashboard\n"
        f"Exec=bash -c 'cd \"{project_dir}\" && \"{streamlit}\" run app.py'\n"
        f"Path={project_dir}\n"
        f"Terminal=true\n"
        f"Categories=Finance;\n",
        encoding="utf-8",
    )
    desktop_file.chmod(0o755)
    _ok(f"Launcher created: {desktop_file}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  TradeAgent Setup")
    print("=" * 60)

    env = _read_env()
    all_updates: dict[str, str] = {}

    all_updates.update(check_anthropic(env))
    all_updates.update(check_itradedash(env))
    all_updates.update(check_schwab(env))

    if all_updates:
        _write_env(all_updates)
        _ok(f".env updated ({', '.join(all_updates.keys())})")

    init_database()
    import_bundle()

    print(f"\n{SEP}")
    print("6. Insider Intel Connection Check")
    print(SEP)
    verify_insider()

    create_desktop_shortcut()

    print(f"\n{'=' * 60}")
    print("  Setup complete.")
    print("  Run:  streamlit run app.py")
    print("  Or double-click the TradeAgent shortcut on your Desktop.")
    print("=" * 60)


if __name__ == "__main__":
    main()
