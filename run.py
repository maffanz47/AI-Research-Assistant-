"""
run.py
======
Single-command runner for the Research Gap Finder FastAPI backend.

What it does
------------
1. Starts Uvicorn serving api:app on http://127.0.0.1:8000
2. Launches a reverse SSH tunnel (Pinggy -> localhost.run -> Serveo)
   that exposes port 8000 publicly over HTTPS
3. Parses tunnel output, strips ANSI colors, and prints the public HTTPS URL
4. On Ctrl+C: cleanly terminates BOTH Uvicorn and the SSH tunnel

Usage
-----
  python run.py                     # Mock LLM mode
  USE_MOCK_LLM=false python run.py # Real Qwen LLM mode
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
# Configuration
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 8000
APP_MODULE = "api:app"
WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))
UVICORN_STARTUP_WAIT = 3

# ---------------------------------------------------------------------------
# Tunnel providers (pinggy -> localhost.run -> serveo)
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

# Regex patterns to extract URLs
_URL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(https://[a-zA-Z0-9\-]+\.a\.pinggy\.[a-z]+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.pinggy\.[a-z]+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.lhr\.life)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-]+\.serveo\.net)", re.IGNORECASE),
    re.compile(r"Forwarding\s+(https?://\S+)", re.IGNORECASE),
    re.compile(r"(https://[a-zA-Z0-9\-\.]+\.[a-z]{2,})", re.IGNORECASE),
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# ---------------------------------------------------------------------------
# ANSI Terminal Formatting Helpers
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


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for reliable regex matching."""
    return _ANSI_RE.sub("", text)


def _print_banner() -> None:
    print()
    print(_c(_BOLD + _CYAN, "======================================================"))
    print(_c(_BOLD + _CYAN, "       Research Gap Finder — Deployment Runner        "))
    print(_c(_BOLD + _CYAN, "======================================================"))
    print()
    use_mock = os.getenv("USE_MOCK_LLM", "true").strip().lower() not in ("false", "0", "no")
    model    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    print(f"  {'LLM mode':<18}: {_c(_YELLOW, 'MOCK') if use_mock else _c(_GREEN, f'REAL  ({model})')}")
    print(f"  {'Uvicorn target':<18}: {APP_MODULE}  (workers={WORKERS})")
    print(f"  {'Local API':<18}: http://{HOST}:{PORT}")
    print()


# ---------------------------------------------------------------------------
# Process Management & Signal Handlers
# ---------------------------------------------------------------------------

_procs: list[subprocess.Popen] = []


def _terminate_all() -> None:
    """Send SIGTERM (or taskkill on Windows) to every managed subprocess."""
    for proc in _procs:
        if proc.poll() is None:
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
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
# Uvicorn Process Launcher
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
# Tunnel Launcher & Output Stream Reader
# ---------------------------------------------------------------------------

def _stream_tunnel_output(proc: subprocess.Popen, found_event: threading.Event, provider_name: str) -> None:
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
                    if any(ignored in public_url for ignored in ["github.com", "pinggy.io/docs", "localhost"]):
                        continue
                    url_found = True
                    found_event.set()
                    _print_public_url(public_url)
                    break


def _print_public_url(url: str) -> None:
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
                stderr=subprocess.STDOUT,
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
# Main Execution Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGINT,  _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    _print_banner()

    # 1. Start Uvicorn
    uvicorn_proc = _start_uvicorn()
    print(_c(_YELLOW, f"  Waiting {UVICORN_STARTUP_WAIT}s for Uvicorn to initialise…"))
    time.sleep(UVICORN_STARTUP_WAIT)

    if uvicorn_proc.poll() is not None:
        print(_c(_RED, f"✗  Uvicorn exited early (code {uvicorn_proc.returncode})."))
        sys.exit(1)

    print(_c(_GREEN, f"✓  Uvicorn is up at http://{HOST}:{PORT}"))
    print()

    # 2. Start SSH reverse tunnel
    _start_tunnel()

    # 3. Keep running until Ctrl+C
    try:
        uvicorn_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _terminate_all()


if __name__ == "__main__":
    main()
