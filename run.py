"""
run.py
======
Single-command runner for the Research Gap Finder FastAPI backend.

What it does
------------
1. Starts Uvicorn serving api:app on http://127.0.0.1:8000
2. Launches a Pinggy SSH reverse tunnel that exposes port 8000 publicly
   over HTTPS (no account or token required for the free tier)
3. Parses the tunnel output and prints the public HTTPS URL clearly so
   you can copy it straight into your frontend config or share it
4. On Ctrl+C: gracefully terminates BOTH Uvicorn and the SSH tunnel

Usage
-----
  # On the remote GPU machine (after git pull):
  python run.py

  # Switch to real Qwen model:
  USE_MOCK_LLM=false python run.py

  # Custom Uvicorn workers:
  UVICORN_WORKERS=2 python run.py

Zero-config — reads USE_MOCK_LLM / OLLAMA_MODEL / OLLAMA_BASE_URL
from the environment; no .env file required (but supported if present).

Dependencies: standard library only (subprocess, threading, re, time, os,
              signal, sys, shlex).
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"          # Uvicorn binds locally; tunnel exposes it
PORT = 8000
APP_MODULE = "api:app"

# Number of Uvicorn worker processes. Keep 1 for GPU machines
# (model is loaded once; multiple workers would duplicate VRAM usage).
WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))

# Pinggy free SSH tunnel command — no registration needed
PINGGY_CMD = (
    "ssh -p 443 "
    "-o StrictHostKeyChecking=no "
    "-o ServerAliveInterval=30 "
    "-o ExitOnForwardFailure=yes "
    f"-R0:localhost:{PORT} "
    "free.pinggy.io"
)

# Regex patterns to detect the public URL from Pinggy's stdout/stderr
# Pinggy prints something like:
#   https://rAnDomStr.a.pinggy.io
#   or:
#   Forwarding  https://xxxx.a.pinggy.link -> localhost:8000
_URL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(https://[a-zA-Z0-9\-]+\.a\.pinggy\.[a-z]+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.pinggy\.[a-z]+)", re.IGNORECASE),
    re.compile(r"Forwarding\s+(https://\S+)", re.IGNORECASE),
    # Generic fallback — any https URL that appeared in tunnel output
    re.compile(r"(https://\S+)", re.IGNORECASE),
]

# How long to wait (seconds) for Uvicorn to be ready before launching tunnel
UVICORN_STARTUP_WAIT = 3

# ---------------------------------------------------------------------------
# ANSI colours for nicer console output
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"


def _c(color: str, text: str) -> str:
    """Wrap text in ANSI colour if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print()
    print(_c(_BOLD + _CYAN, "╔══════════════════════════════════════════════════════╗"))
    print(_c(_BOLD + _CYAN, "║       Research Gap Finder — Deployment Runner        ║"))
    print(_c(_BOLD + _CYAN, "╚══════════════════════════════════════════════════════╝"))
    print()
    use_mock = os.getenv("USE_MOCK_LLM", "true").strip().lower() not in ("false", "0", "no")
    model    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    print(f"  {'LLM mode':<18}: {_c(_YELLOW, 'MOCK') if use_mock else _c(_GREEN, f'REAL  ({model})')}")
    print(f"  {'Uvicorn target':<18}: {APP_MODULE}  (workers={WORKERS})")
    print(f"  {'Local API':<18}: http://{HOST}:{PORT}")
    print()


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_procs: list[subprocess.Popen] = []


def _terminate_all() -> None:
    """Send SIGTERM (or taskkill on Windows) to every managed subprocess."""
    for proc in _procs:
        if proc.poll() is None:  # still running
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
    # Give them 3 s to exit, then SIGKILL
    deadline = time.time() + 3
    for proc in _procs:
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def _signal_handler(signum: int, _frame: object) -> None:
    print()
    print(_c(_YELLOW, "\n⏹  Shutting down (received signal %d)…" % signum))
    _terminate_all()
    print(_c(_GREEN, "✅  All processes stopped. Goodbye!"))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Uvicorn launcher
# ---------------------------------------------------------------------------

def _start_uvicorn() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn",
        APP_MODULE,
        "--host", HOST,
        "--port", str(PORT),
        "--workers", str(WORKERS),
    ]
    print(_c(_CYAN, f"▶  Starting Uvicorn: {' '.join(cmd)}"))
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    _procs.append(proc)
    return proc


# ---------------------------------------------------------------------------
# Pinggy SSH tunnel launcher + URL extractor
# ---------------------------------------------------------------------------

def _stream_tunnel_output(proc: subprocess.Popen, found_event: threading.Event) -> None:
    """
    Background thread: read tunnel stdout/stderr line by line.
    When we spot a public HTTPS URL, print it prominently and set the event.
    Also relay every line to the console so the user can see tunnel status.
    """
    url_found = False
    # Pinggy writes the URL to stderr in most versions
    for line in proc.stderr:  # type: ignore[union-attr]
        line = line.rstrip()
        print(_c(_YELLOW, f"  [tunnel] {line}"))
        if not url_found:
            for pattern in _URL_PATTERNS:
                m = pattern.search(line)
                if m:
                    public_url = m.group(1).rstrip("/")
                    url_found = True
                    found_event.set()
                    _print_public_url(public_url)
                    break


def _print_public_url(url: str) -> None:
    """Print the public URL in a prominent box."""
    print()
    print(_c(_BOLD + _GREEN, "┌─────────────────────────────────────────────────────┐"))
    print(_c(_BOLD + _GREEN, "│                                                     │"))
    print(_c(_BOLD + _GREEN, "│   🌐  Public HTTPS URL (copy this!)                 │"))
    print(_c(_BOLD + _GREEN, f"│   {url:<51} │"))
    print(_c(_BOLD + _GREEN, "│                                                     │"))
    print(_c(_BOLD + _GREEN, "│   Endpoints:                                        │"))
    print(_c(_BOLD + _GREEN, f"│   GET  {url}/ ← status + GPU flag  │"))
    print(_c(_BOLD + _GREEN, f"│   POST {url}/infer ← abstract → gaps  │"))
    print(_c(_BOLD + _GREEN, f"│   POST {url}/analyze ← full pipeline  │"))
    print(_c(_BOLD + _GREEN, "│                                                     │"))
    print(_c(_BOLD + _GREEN, "│   Press Ctrl+C to stop both processes               │"))
    print(_c(_BOLD + _GREEN, "└─────────────────────────────────────────────────────┘"))
    print()


def _start_tunnel() -> subprocess.Popen:
    print(_c(_CYAN, f"▶  Opening Pinggy tunnel: {PINGGY_CMD[:60]}…"))
    proc = subprocess.Popen(
        shlex.split(PINGGY_CMD),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,        # line-buffered
    )
    _procs.append(proc)

    # Thread to capture URL from tunnel stderr
    found_event = threading.Event()
    t = threading.Thread(
        target=_stream_tunnel_output,
        args=(proc, found_event),
        daemon=True,
    )
    t.start()

    # Wait up to 20 s for the URL to appear
    if not found_event.wait(timeout=20):
        print(_c(_YELLOW,
            "  ⚠  Pinggy URL not detected within 20 s — "
            "check tunnel output above or look for 'https://' lines."
        ))

    return proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Register Ctrl+C / SIGTERM handlers
    signal.signal(signal.SIGINT,  _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    _print_banner()

    # 1. Start Uvicorn
    uvicorn_proc = _start_uvicorn()
    print(_c(_YELLOW, f"  Waiting {UVICORN_STARTUP_WAIT}s for Uvicorn to initialise (model load)…"))
    time.sleep(UVICORN_STARTUP_WAIT)

    # Check Uvicorn didn't die immediately (e.g. port conflict)
    if uvicorn_proc.poll() is not None:
        print(_c(_RED, f"✗  Uvicorn exited early (code {uvicorn_proc.returncode})."))
        print(_c(_RED,  "   Check port conflicts: lsof -i :8000"))
        sys.exit(1)

    print(_c(_GREEN, f"✓  Uvicorn is up at http://{HOST}:{PORT}"))
    print()

    # 2. Start Pinggy tunnel
    _start_tunnel()

    # 3. Block until Uvicorn exits naturally or user hits Ctrl+C
    try:
        uvicorn_proc.wait()
    except KeyboardInterrupt:
        pass  # signal handler will clean up
    finally:
        _terminate_all()


if __name__ == "__main__":
    main()
