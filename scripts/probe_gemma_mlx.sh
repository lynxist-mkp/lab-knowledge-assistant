#!/usr/bin/env bash
# Probe mlx_vlm.server with ModelScope-downloaded Gemma 4 E2B for text + image caption.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE_DIR="$ROOT/.scratch/gemma-probe"
VENV="$ROOT/.venvs/paddleocr-env"
PY="$VENV/bin/python"
MODEL_ID="mlx-community/gemma-4-e2b-it-mxfp4"
PORT="${GEMMA_PROBE_PORT:-8120}"
SERVER_URL="http://127.0.0.1:${PORT}/"
IMG="$PROBE_DIR/test-image.png"

mkdir -p "$PROBE_DIR"

if [[ ! -f "$IMG" ]]; then
  "$ROOT/.venv/bin/python" - <<PY
from pathlib import Path
from PIL import Image

out = Path("$IMG")
img = Image.new("RGB", (320, 240), color=(180, 40, 40))
for x in range(40, 280):
    for y in range(80, 120):
        img.putpixel((x, y), (240, 220, 60))
img.save(out)
print(out)
PY
fi

echo "Resolving ModelScope model: ${MODEL_ID}"
MODEL_PATH="$("$PY" "$ROOT/scripts/resolve_modelscope_model.py" "$MODEL_ID")"
echo "Local path: ${MODEL_PATH}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting mlx_vlm.server on port ${PORT}..."
"$PY" -m mlx_vlm.server --port "${PORT}" --model "${MODEL_PATH}" &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -sf "${SERVER_URL}health" >/dev/null; then
    echo "Server ready."
    break
  fi
  sleep 1
done

curl -sf "${SERVER_URL}health" >/dev/null

echo
echo "=== Text generation ==="
curl -s "${SERVER_URL}v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<EOF
{
  "model": "${MODEL_ID}",
  "messages": [{"role": "user", "content": "用一句话说明湄洲妈祖祖庙在妈祖信仰中的地位。"}],
  "max_tokens": 120,
  "temperature": 0.0
}
EOF
)" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"

echo
echo "=== Image caption ==="
IMG_B64="$("$PY" - <<PY
import base64
from pathlib import Path
data = Path("$IMG").read_bytes()
print(base64.b64encode(data).decode())
PY
)"
curl -s "${SERVER_URL}v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<EOF
{
  "model": "${MODEL_ID}",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${IMG_B64}"}},
      {"type": "text", "text": "请用中文客观描述这张图片的主要颜色和内容，不超过两句话。"}
    ]
  }],
  "max_tokens": 120,
  "temperature": 0.0
}
EOF
)" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"

echo
echo "Probe complete."
