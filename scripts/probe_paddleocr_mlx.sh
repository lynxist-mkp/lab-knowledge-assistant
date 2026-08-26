#!/usr/bin/env bash
# Probe mlx-vlm.server with ModelScope-downloaded PaddleOCR-VL weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venvs/paddleocr-env"
PORT="${PADDLEOCR_MLX_PORT:-8111}"
PRIMARY_MODEL="${PADDLEOCR_MLX_MODEL:-mlx-community/PaddleOCR-VL-1.6-5bit}"
FALLBACK_MODEL="${PADDLEOCR_MLX_FALLBACK:-PaddlePaddle/PaddleOCR-VL-1.6}"
PROBE_PDF="${PROBE_PDF:-}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing ${VENV}. Run: ${ROOT}/scripts/setup_paddleocr_env.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

resolve_model() {
  python "${ROOT}/scripts/resolve_modelscope_model.py" "$1"
}

if [[ -z "${PROBE_PDF}" ]]; then
  PROBE_PDF="$(mktemp /tmp/wenmai-scan-probe.XXXXXX.pdf)"
  python - <<'PY' "${PROBE_PDF}"
import sys
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

out = Path(sys.argv[1])
image_path = out.with_suffix(".png")
Image.new("RGB", (80, 80), color=(20, 80, 140)).save(image_path)
pdf = canvas.Canvas(str(out), pagesize=letter)
pdf.drawImage(str(image_path), 72, 580, width=160, height=160)
pdf.save()
PY
  trap 'rm -f "${PROBE_PDF}" "${PROBE_PDF%.pdf}.png"' EXIT
fi

start_server() {
  local model_path="$1"
  pkill -f "mlx_vlm.server.*--port ${PORT}" 2>/dev/null || true
  sleep 1
  mlx_vlm.server --port "${PORT}" --model "${model_path}" &
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "mlx_vlm.server ready on port ${PORT} with ${model_path}"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "mlx_vlm.server exited early for ${model_path}" >&2
      return 1
    fi
    sleep 1
  done
  kill "${SERVER_PID}" 2>/dev/null || true
  echo "mlx_vlm.server timeout for ${model_path}" >&2
  return 1
}

run_parse() {
  local model_path="$1"
  local out_json
  out_json="$(mktemp "${TMPDIR:-/tmp}/paddleocr-probe-out.XXXXXX")"
  out_json="${out_json}.json"
  python "${ROOT}/scripts/paddleocr_parse_pdf.py" \
    "${PROBE_PDF}" \
    -o "${out_json}" \
    --vl-rec-backend mlx-vlm-server \
    --vl-rec-server-url "http://127.0.0.1:${PORT}/" \
    --vl-rec-api-model-name "${model_path}"
  echo "Parse OK (${model_path}). Pages: $(python -c "import json; print(len(json.load(open('${out_json}'))['pages']))")"
  rm -f "${out_json}"
}

echo "Resolving primary from ModelScope: ${PRIMARY_MODEL}"
PRIMARY_PATH="$(resolve_model "${PRIMARY_MODEL}")"
ACTIVE_PATH="${PRIMARY_PATH}"

if start_server "${PRIMARY_PATH}"; then
  run_parse "${PRIMARY_PATH}"
else
  echo "Primary failed; trying fallback ${FALLBACK_MODEL}" >&2
  FALLBACK_PATH="$(resolve_model "${FALLBACK_MODEL}")"
  ACTIVE_PATH="${FALLBACK_PATH}"
  start_server "${FALLBACK_PATH}"
  run_parse "${FALLBACK_PATH}"
fi

kill "${SERVER_PID}" 2>/dev/null || true
echo "Probe succeeded with local model: ${ACTIVE_PATH}"
