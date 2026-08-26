#!/usr/bin/env bash
# Isolated PaddleOCR-VL + mlx-vlm env for ticket 33 (not installed in main RAG venv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venvs/paddleocr-env"

if [[ ! -d "${VENV}" ]]; then
  python3.12 -m venv "${VENV}"
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"
pip install -U pip
pip install "paddleocr>=3.0" "paddlepaddle" "mlx-vlm>=0.3.11" "paddlex[ocr]==3.7.2" "openai>=1.63" pillow reportlab

echo "PaddleOCR env ready: ${VENV}"
echo "Probe: ${ROOT}/scripts/probe_paddleocr_mlx.sh"
