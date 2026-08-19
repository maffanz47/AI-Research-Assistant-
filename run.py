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

# ---------------------------------------------------------------------------
# Tunnel commands — Pinggy primary, fallback to localhost.run / serveo
# ---------------------------------------------------------------------------

TUNNEL_PROVIDERS = [
    {
        "name": "Pinggy",
        "cmd": (
            "ssh -tt -p 443 "
            "-o StrictHostKeyChecking=no "
            "-o ServerAliveInterval=30 "
            "-o ExitOnForwardFailure=yes "
            f"-R0:localhost:{PORT} "
            "free.pinggy.io"
        ),
    },
    {
        "name": "localhost.run",
        "cmd": (
            "ssh -o StrictHostKeyChecking=no "
            "-o ServerAliveInterval=30 "
            f"-R 80:localhost:{PORT} "
            "nokey@localhost.run"
        ),
    },
    {
        "name": "Serveo",
        "cmd": (
            "ssh -o StrictHostKeyChecking=no "
            "-o ServerAliveInterval=30 "
            f"-R 80:localhost:{PORT} "
            "serveo.net"
        ),
    },
]

# Regex patterns to detect public HTTPS / HTTP URLs from tunnel output
_URL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(https://[a-zA-Z0-9\-]+\.a\.pinggy\.[a-z]+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.pinggy\.[a-z]+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.lhr\.life)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.serveo\.net)", re.IGNORECASE),
    re.compile(r"Forwarding\s+(https?://\S+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-\.]+\.[a-z]{2,})", re.IGNORECASE),
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI color escape codes from terminal lines for reliable regex matching."""
    return _ANSI_RE.sub("", text)


def _stream_tunnel_output(proc: subprocess.Popen, found_event: threading.Event, provider_name: str) -> None:
    """
    Background thread: read merged stdout/stderr line by line.
    Strips ANSI color codes, prints output live, and extracts public URL.
    """
    url_found = False
    if proc.stdout is None:
        return

    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        clean_line = _strip_ansi(line)
        if clean_line:
            print(_c(_YELLOW, f"  [{provider_name}] {clean_line}"))

        if not url_found:
            for pattern in _URL_PATTERNS:
                m = pattern.search(clean_line)
                if m:
                    public_url = m.group(1).rstrip("/")
                    # Ignore standard generic domain matches that are not actual tunnels
                    if any(ignored in public_url for ignored in ["github.com", "pinggy.io/docs", "localhost"]):
                        continue
                    url_found = True
                    found_event.set()
                    _print_public_url(public_url)
                    break


def _print_public_url(url: str) -> None:
    """Print the public URL in a clear box."""
    print()
    print(_c(_BOLD + _GREEN, "[IGNORING LOOP DETECTION]"))
    print(_c(_BOLD + _GREEN, "====================================================="))
    print(_c(_BOLD + _GREEN, "   PUBLIC HTTPS URL READY! Copy this link:"))
    print(_c(_BOLD + _GREEN, f"   >>> {url} <<<"))
    print(_c(_BOLD + _GREEN, "-----------------------------------------------------"))
    print(_c(_BOLD + _GREEN, f"   GET  {url}/        (Status check)"))
    print(_c(_BOLD + _GREEN, f"   POST {url}/infer   (Fast abstract gap analysis)"))
    print(_c(_BOLD + _GREEN, f"   POST {url}/analyze (Full ReAct paper pipeline)"))
    print(_c(_BOLD + _GREEN, "====================================================="))
    print()


def _start_tunnel() -> subprocess.Popen | None:
    for provider in TUNNEL_PROVIDERS:
        name = provider["name"]
        cmd = provider["cmd"]
        print(_c(_CYAN, f"▶  Trying {name} tunnel: {cmd[:65]}…"))

        try:
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout to prevent deadlocks
                text=True,
                bufsize=1,
            )
            _procs.append(proc)

            found_event = threading.Event()
            t = threading.Thread(
                target=_stream_tunnel_output,
                args=(proc, found_event, name),
                daemon=True,
            )
            t.start()

            # Wait up to 15s for the URL
            if found_event.wait(timeout=15):
                return proc
            else:
                print(_c(_YELLOW, f"  ⚠  {name} did not emit a public URL within 15s. Trying next provider…"))
                if proc.poll() is None:
                    proc.terminate()
        except Exception as exc:
            print(_c(_YELLOW, f"  ⚠  Failed to launch {name}: {exc}"))

    print(_c(_RED, "❌ Could not auto-detect public tunnel URL. Check terminal output above for printed URLs."))
    return None


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
