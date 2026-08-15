#!/usr/bin/env bash
# gpu_setup.sh
# ============
# Provision a remote Ubuntu/Debian GPU machine (24 GB VRAM) for the
# Research Gap Finder LLM inference pipeline.
#
# What this script does:
#   1. Updates system packages and installs curl
#   2. Installs Ollama from the official one-liner
#   3. Starts the Ollama background service
#   4. Pulls the quantised Qwen 2.5-14B instruct model (Q4_K_M, ~9 GB)
#   5. Verifies the model is available and shows GPU VRAM info
#
# Usage:
#   chmod +x gpu_setup.sh
#   ./gpu_setup.sh
#
# After running, start the FastAPI backend with:
#   USE_MOCK_LLM=false OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M \
#     uvicorn api:app --host 0.0.0.0 --port 8000

set -euo pipefail

OLLAMA_MODEL_TAG="qwen2.5:14b-instruct-q4_K_M"
OLLAMA_SERVICE_PORT=11434

echo "============================================================"
echo "  Research Gap Finder — GPU Machine Setup"
echo "  Model: $OLLAMA_MODEL_TAG"
echo "============================================================"

# ── 1. System dependencies ─────────────────────────────────────────
echo "[1/5] Updating package index and installing prerequisites …"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends curl ca-certificates lsof

# ── 2. Install Ollama ─────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    echo "[2/5] Ollama already installed: $(ollama --version)"
else
    echo "[2/5] Installing Ollama …"
    curl -fsSL https://ollama.com/install.sh | sh
    echo "      Ollama installed: $(ollama --version)"
fi

# ── 3. Start Ollama service ───────────────────────────────────────
echo "[3/5] Starting Ollama service in background …"

# Kill any stale instance first
if lsof -i :"$OLLAMA_SERVICE_PORT" &>/dev/null; then
    echo "      Port $OLLAMA_SERVICE_PORT already in use — assuming Ollama is running."
else
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "      Ollama PID: $OLLAMA_PID  (log: /tmp/ollama.log)"
    # Wait for the REST API to be ready
    echo -n "      Waiting for Ollama API"
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$OLLAMA_SERVICE_PORT/api/version" &>/dev/null; then
            echo " ✓"
            break
        fi
        echo -n "."
        sleep 1
    done
fi

# ── 4. Pull Qwen 2.5-14B (Q4_K_M quantisation) ───────────────────
echo "[4/5] Pulling model: $OLLAMA_MODEL_TAG (this may take several minutes) …"
ollama pull "$OLLAMA_MODEL_TAG"
echo "      Model pulled successfully."

# ── 5. Verify & display GPU info ─────────────────────────────────
echo "[5/5] Verification …"
echo ""
echo "  Available Ollama models:"
ollama list
echo ""
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU VRAM status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "  nvidia-smi not found — check GPU driver installation."
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To run inference:"
echo "    export USE_MOCK_LLM=false"
echo "    export OLLAMA_MODEL=$OLLAMA_MODEL_TAG"
echo "    uvicorn api:app --host 0.0.0.0 --port 8000"
echo ""
echo "  To fine-tune:"
echo "    python train.py --dataset_dir cache/ --output_dir lora_adapter/"
echo "============================================================"
