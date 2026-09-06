#!/usr/bin/env bash
# Stop local uvicorn and mlx_vlm servers (Gemma :8120, PaddleOCR :8111).
set -euo pipefail

GEMMA_PORT="${LAB_KNOWLEDGE_GEMMA_PORT:-${LAB_KNOWLEDGE_GEMMA_PORT:-8120}}"
PADDLEOCR_PORT="${LAB_KNOWLEDGE_LAB_KNOWLEDGE_PADDLEOCR_MLX_PORT:-${LAB_KNOWLEDGE_PADDLEOCR_MLX_PORT:-8111}}"

kill_pids() {
  local label="$1"
  shift
  local pid
  for pid in "$@"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Killing ${label}: pid=${pid}"
      kill "$pid" 2>/dev/null || true
    fi
  done
}

collect_pgrep() {
  local pattern="$1"
  pgrep -f "$pattern" 2>/dev/null || true
}

collect_port_pids() {
  local port="$1"
  local pattern="$2"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti:"${port}" 2>/dev/null || true)"
  fi
  if [[ -z "$pids" ]]; then
    pids="$(collect_pgrep "${pattern}")"
  fi
  echo "$pids"
}

kill_by_pattern() {
  local label="$1"
  local pattern="$2"
  local pids
  pids="$(collect_pgrep "${pattern}")"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill_pids "${label}" ${pids}
  fi
}

kill_mlx_port() {
  local label="$1"
  local port="$2"
  local pids
  pids="$(collect_port_pids "${port}" "mlx_vlm.server.*--port ${port}")"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill_pids "${label}" ${pids}
  fi
}

kill_by_pattern "uvicorn (create_app)" "uvicorn (wenmai|lab_knowledge)\.app:create_app"
kill_by_pattern "uvicorn (app)" "uvicorn (wenmai|lab_knowledge)\.app:app"
kill_mlx_port "mlx_vlm.server (Gemma)" "${GEMMA_PORT}"
kill_mlx_port "mlx_vlm.server (PaddleOCR)" "${PADDLEOCR_PORT}"

exit 0
