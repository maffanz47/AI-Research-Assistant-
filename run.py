"""
run.py
======
Single-command runner for the Research Gap Finder FastAPI backend.

Starts Uvicorn on port 8000, then opens an SSH reverse tunnel via
localhost.run which provides a stable public HTTPS URL — no account,
no installation, no firewall rules required.

Usage
-----
  python run.py                       # Mock LLM (default, for testing)
  USE_MOCK_LLM=false python run.py    # Real Qwen model via Ollama
  UVICORN_WORKERS=2 python run.py     # Multiple workers
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST    = os.getenv("UVICORN_HOST", "0.0.0.0")
PORT    = int(os.getenv("UVICORN_PORT", "8000"))
MODULE  = "api:app"
WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))

# SSH tunnel command — localhost.run; no account needed
TUNNEL_CMD = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=5",
    "-o", "ExitOnForwardFailure=yes",
    "-R", f"80:localhost:{PORT}",
    "nokey@localhost.run",
    "--",
    "--output=json",
]

# ---------------------------------------------------------------------------
# ANSI helpers (ASCII-only, safe on all terminals)
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}" if sys.stdout.isatty() else text

def _banner() -> None:
    use_mock = os.getenv("USE_MOCK_LLM", "true").strip().lower() not in ("false", "0", "no")
    model    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    print()
    print(_c(_BOLD + _CYAN, "=" * 56))
    print(_c(_BOLD + _CYAN, "     Research Gap Finder - Backend Runner"))
    print(_c(_BOLD + _CYAN, "=" * 56))
    print(f"  LLM mode : {_c(_YELLOW, 'MOCK') if use_mock else _c(_GREEN, 'REAL (' + model + ')')}")
    print(f"  Local    : http://localhost:{PORT}")
    print(_c(_CYAN, "  Starting SSH tunnel to localhost.run..."))
    print()

# ---------------------------------------------------------------------------
# Process references (module-level so signal handler can reach them)
# ---------------------------------------------------------------------------
_uvicorn_proc: subprocess.Popen | None = None
_tunnel_proc:  subprocess.Popen | None = None


def _kill_all() -> None:
    for p in (_tunnel_proc, _uvicorn_proc):
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=4)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def _signal_handler(signum: int, _frame: object) -> None:
    print()
    print(_c(_YELLOW, "Shutting down... (Ctrl+C received)"))
    _kill_all()
    print(_c(_GREEN, "All processes stopped. Goodbye!"))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Tunnel URL reader (runs in a background thread)
# ---------------------------------------------------------------------------
import re

_DOMAIN_RE = re.compile(r'"domain"\s*:\s*"([^"]+\.lhr\.life)"')

def _read_tunnel_output(proc: subprocess.Popen) -> None:
    """Read tunnel output line-by-line and print the HTTPS URL when found."""
    assert proc.stdout is not None
    url_printed = False
    for raw in proc.stdout:
        line = raw.strip()
        if not url_printed:
            m = _DOMAIN_RE.search(line)
            if m:
                url = f"https://{m.group(1)}"
                url_printed = True
                print()
                print(_c(_BOLD + _GREEN, "=" * 56))
                print(_c(_BOLD + _GREEN, "  PUBLIC HTTPS URL READY"))
                print(_c(_BOLD + _GREEN, "=" * 56))
                print()
                print(_c(_BOLD + _GREEN, f"  >>> {url}"))
                print()
                print(_c(_GREEN,  "  Paste this URL into your Vercel frontend."))
                print(_c(_YELLOW, "  It changes each time the server restarts."))
                print()
                print(_c(_CYAN,  "  Press Ctrl+C to stop the server."))
                print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global _uvicorn_proc, _tunnel_proc

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    _banner()

    # 1. Start Uvicorn
    uv_cmd = [
        sys.executable, "-m", "uvicorn", MODULE,
        "--host", HOST,
        "--port", str(PORT),
        "--workers", str(WORKERS),
    ]
    _uvicorn_proc = subprocess.Popen(
        uv_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        stdin=subprocess.DEVNULL,
    )

    # Wait and check it didn't immediately crash
    time.sleep(2)
    if _uvicorn_proc.poll() is not None:
        print(_c(_RED, f"Uvicorn exited immediately (code {_uvicorn_proc.returncode})."))
        sys.exit(1)

    print(_c(_GREEN, f"Uvicorn running at http://localhost:{PORT}"))
    print(_c(_CYAN, "Opening SSH tunnel to localhost.run ..."))

    # 2. Start SSH tunnel
    _tunnel_proc = subprocess.Popen(
        TUNNEL_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    # 3. Parse tunnel URL in background thread
    t = threading.Thread(target=_read_tunnel_output, args=(_tunnel_proc,), daemon=True)
    t.start()

    # 4. Block — restart tunnel if it dies while uvicorn is still alive
    while True:
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            break

        # If uvicorn died, exit everything
        if _uvicorn_proc.poll() is not None:
            print(_c(_RED, "Uvicorn stopped unexpectedly. Exiting."))
            _kill_all()
            sys.exit(1)

        # If tunnel died, restart it
        if _tunnel_proc.poll() is not None:
            print(_c(_YELLOW, "Tunnel closed. Restarting in 3s..."))
            time.sleep(3)
            _tunnel_proc = subprocess.Popen(
                TUNNEL_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            t2 = threading.Thread(target=_read_tunnel_output, args=(_tunnel_proc,), daemon=True)
            t2.start()


if __name__ == "__main__":
    main()
