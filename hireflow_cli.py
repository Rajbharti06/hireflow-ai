"""
hireflow — global CLI launcher for HireFlow AI.

Installed as a console-script entry point via:
    pip install -e .

Usage:
    hireflow              # start on default port 8501
    hireflow --port 8502  # start on a custom port
    hireflow --no-browser # skip auto-opening the browser
    hireflow --stop       # kill any running HireFlow server
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path


# ── Project root = the directory that contains this file ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
APP_PY       = PROJECT_ROOT / "app.py"
ENV_FILE     = PROJECT_ROOT / ".env"
VENV_DIR     = PROJECT_ROOT / ".venv"


def _find_streamlit() -> Path:
    """
    Prefer the project venv's streamlit so the right deps are used.
    Falls back to any streamlit on PATH.
    """
    candidates = [
        VENV_DIR / "Scripts" / "streamlit.exe",   # Windows venv
        VENV_DIR / "Scripts" / "streamlit",        # Windows venv (no ext)
        VENV_DIR / "bin"     / "streamlit",        # Unix venv
    ]
    for c in candidates:
        if c.exists():
            return c

    # Last resort: whatever `streamlit` resolves to on PATH
    import shutil
    found = shutil.which("streamlit")
    if found:
        return Path(found)

    sys.exit(
        "\n  [ERROR] streamlit not found.\n"
        "  Run:  pip install -r requirements.txt\n"
        "  from the project directory, then retry.\n"
    )


def _load_env():
    """Load .env into the current process environment."""
    if not ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
    except ImportError:
        # python-dotenv not installed in system Python — parse manually
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val


def _open_browser(url: str, delay: float = 2.5):
    """Open the browser after a short delay (non-blocking)."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def _banner(port: int):
    print()
    print("  ==========================================")
    print("    HireFlow AI  |  Resume Screener")
    print("  ==========================================")
    print(f"  >> http://localhost:{port}")
    print("  >> Ctrl+C to stop\n")


def _stop():
    """Kill any streamlit process running app.py on this machine."""
    import signal
    killed = 0
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "streamlit" in cmdline and "app.py" in cmdline:
                proc.send_signal(signal.SIGTERM)
                killed += 1
    except ImportError:
        # psutil not available — try taskkill on Windows
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "streamlit.exe"],
            capture_output=True, text=True
        )
        if "SUCCESS" in result.stdout:
            killed = 1

    if killed:
        print(f"  Stopped {killed} HireFlow server(s).")
    else:
        print("  No running HireFlow server found.")


def main():
    parser = argparse.ArgumentParser(
        prog="hireflow",
        description="Launch HireFlow AI resume screener",
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Port to serve on (default: 8501)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't open the browser automatically"
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop any running HireFlow server"
    )
    args = parser.parse_args()

    if args.stop:
        _stop()
        return

    if not APP_PY.exists():
        sys.exit(
            f"\n  [ERROR] app.py not found at {PROJECT_ROOT}\n"
            "  Make sure you installed hireflow-ai from the correct directory.\n"
        )

    # Load env vars before launching so child process inherits them
    _load_env()

    streamlit = _find_streamlit()
    url       = f"http://localhost:{args.port}"

    _banner(args.port)

    if not args.no_browser:
        _open_browser(url, delay=3.0)

    cmd = [
        str(streamlit), "run", str(APP_PY),
        "--server.port",     str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n  HireFlow stopped.")


if __name__ == "__main__":
    main()
