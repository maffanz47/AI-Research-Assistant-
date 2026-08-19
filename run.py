"""
run.py
======
Single-command runner for the Research Gap Finder FastAPI backend.

Starts Uvicorn on 0.0.0.0:8000 (accessible over the network).
No tunnel, no SSH — just run and forget.

Usage
-----
  python run.py                       # Mock LLM mode
  USE_MOCK_LLM=false python run.py    # Real Qwen model via Ollama
  UVICORN_WORKERS=2 python run.py     # Multiple workers (not recommended for GPU)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = os.getenv("UVICORN_HOST", "0.0.0.0")   # Bind to all interfaces
PORT = int(os.getenv("UVICORN_PORT", "8000"))
APP_MODULE = "api:app"
WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))

# ---------------------------------------------------------------------------
# ANSI Terminal Helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"


def _c(color: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def _print_banner() -> None:
    print()
    print(_c(_BOLD + _CYAN, "======================================================"))
    print(_c(_BOLD + _CYAN, "       Research Gap Finder — Backend Server"))
    print(_c(_BOLD + _CYAN, "======================================================"))
    print()
    use_mock = os.getenv("USE_MOCK_LLM", "true").strip().lower() not in ("false", "0", "no")
    model    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    print(f"  {'LLM mode':<18}: {_c(_YELLOW, 'MOCK') if use_mock else _c(_GREEN, f'REAL  ({model})')}")
    print(f"  {'Uvicorn target':<18}: {APP_MODULE}  (workers={WORKERS})")
    print(f"  {'Listening on':<18}: http://{HOST}:{PORT}")
    print()
    print(_c(_GREEN, "  The server is accessible on the local network."))
    print(_c(_GREEN, f"  Set NEXT_PUBLIC_API_URL=http://<this-machine-ip>:{PORT}"))
    print(_c(_GREEN, "  in your frontend .env.local to connect."))
    print()


# ---------------------------------------------------------------------------
# Process Management
# ---------------------------------------------------------------------------

_proc: subprocess.Popen | None = None


def _signal_handler(signum: int, _frame: object) -> None:
    print()
    print(_c(_YELLOW, "Shutting down (received signal %d)..." % signum))
    if _proc and _proc.poll() is None:
        try:
            if sys.platform == "win32":
                _proc.terminate()
            else:
                _proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    print(_c(_GREEN, "Server stopped. Goodbye!"))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _proc

    signal.signal(signal.SIGINT,  _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    _print_banner()

    cmd = [
        sys.executable, "-m", "uvicorn",
        APP_MODULE,
        "--host", HOST,
        "--port", str(PORT),
        "--workers", str(WORKERS),
    ]
    print(_c(_CYAN, f"Starting: {' '.join(cmd)}"))
    print()

    _proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    # Wait briefly to check for immediate crashes
    time.sleep(2)
    if _proc.poll() is not None:
        print(_c(_RED, f"Uvicorn exited early (code {_proc.returncode})."))
        sys.exit(1)

    print(_c(_GREEN, f"Server running at http://{HOST}:{PORT}"))
    print(_c(_GREEN, "Press Ctrl+C to stop."))
    print()

    # Block until process exits
    try:
        _proc.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
