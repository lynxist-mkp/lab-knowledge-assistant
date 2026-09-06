#!/usr/bin/env bash
# Download PaddleOCR-VL MLX weights from ModelScope (not HuggingFace).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venvs/paddleocr-env"
PRIMARY="${LAB_KNOWLEDGE_PADDLEOCR_MLX_MODEL:-mlx-community/PaddleOCR-VL-1.6-5bit}"
FALLBACK="${LAB_KNOWLEDGE_PADDLEOCR_MLX_FALLBACK:-PaddlePaddle/PaddleOCR-VL-1.6}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing ${VENV}. Run: ${ROOT}/scripts/setup_paddleocr_env.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

resolve() {
  python "${ROOT}/scripts/resolve_modelscope_model.py" "$1"
}

echo "Downloading primary ModelScope model: ${PRIMARY}"
PRIMARY_PATH="$(resolve "${PRIMARY}")"
echo "Primary ready: ${PRIMARY_PATH}"

echo "Downloading fallback ModelScope model: ${FALLBACK}"
FALLBACK_PATH="$(resolve "${FALLBACK}")"
echo "Fallback ready: ${FALLBACK_PATH}"

echo ""
echo "Update settings.yaml paddleocr.mlx_model / mlx_fallback_model to these paths,"
echo "or keep ModelScope ids — runtime resolves them automatically."
