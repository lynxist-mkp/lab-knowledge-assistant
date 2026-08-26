#!/usr/bin/env bash
# Start local Gemma via mlx_vlm.server (needs real Metal — run in Terminal.app, not agent sandbox).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Intact mlx stack lives in the MiniCPM probe venv (paddleocr-env's mlx_vlm was corrupted).
PY="${ROOT}/.scratch/minicpm-probe/.venv/bin/python3.12"
MODEL="${WENMAI_GEMMA_MODEL:-/Users/linkslinks/.cache/modelscope/models/mlx-community--gemma-4-e2b-it-mxfp4/snapshots/master}"
PORT="${WENMAI_GEMMA_PORT:-8120}"
LOG="${ROOT}/.scratch/gemma-server.log"

if [[ ! -x "$PY" ]]; then
  echo "missing python: $PY" >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "missing model at: $MODEL" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
pkill -f "mlx_vlm.server --port ${PORT}" 2>/dev/null || true
sleep 1

echo "Starting mlx_vlm.server on :${PORT}"
echo "  python: $PY"
echo "  model:  $MODEL"
echo "  log:    $LOG"

"$PY" -c 'import mlx.core as mx; print("Metal device:", mx.default_device())'

nohup "$PY" -m mlx_vlm.server --port "$PORT" --model "$MODEL" >"$LOG" 2>&1 &
PID=$!
echo "pid=$PID"

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "READY http://127.0.0.1:${PORT}/health"
    curl -s "http://127.0.0.1:${PORT}/health" || true
    echo
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server died; last log:" >&2
    tail -n 80 "$LOG" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "timeout waiting for health" >&2
tail -n 80 "$LOG" >&2 || true
exit 1
