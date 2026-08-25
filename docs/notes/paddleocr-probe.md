# PaddleOCR-VL / mlx-vlm-server probe (ticket 33)

**Date:** 2026-08-25  
**Goal:** Verify `mlx_vlm.server` can load `mlx-community/PaddleOCR-VL-1.6-5bit` and serve PaddleOCR-VL via `vl_rec_backend="mlx-vlm-server"`.

## Status

Probe **not run** in this implementation session: `.venvs/paddleocr-env` is not present on this machine and `mlx_vlm.server` is not on PATH. Configuration paths in `settings.yaml` point to the expected isolated env location.

## How to probe locally

1. Create isolated env (example):

```bash
python3.12 -m venv .venvs/paddleocr-env
source .venvs/paddleocr-env/bin/activate
pip install "paddleocr>=3.0" "paddlepaddle" "mlx-vlm>=0.3.11" pillow
```

2. Start mlx-vlm.server with 5bit model:

```bash
mlx_vlm.server --port 8111 --model mlx-community/PaddleOCR-VL-1.6-5bit
```

3. In the same env, run a one-page scan through the project script:

```bash
python scripts/paddleocr_parse_pdf.py \
  /path/to/scanned.pdf \
  -o /tmp/paddleocr-out.json \
  --vl-rec-backend mlx-vlm-server \
  --vl-rec-server-url http://127.0.0.1:8111/ \
  --vl-rec-api-model-name mlx-community/PaddleOCR-VL-1.6-5bit
```

4. If step 2 fails to load 5bit, retry with fallback:

```bash
mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6
```

Update `settings.yaml` `paddleocr.mlx_model`, `paddleocr.vl_rec_api_model_name`, and `paddleocr.mlx_fallback_model` accordingly.

## Fallback policy

- 5bit fails → use `PaddlePaddle/PaddleOCR-VL-1.6` on mlx-vlm-server (higher memory).
- OCR parse fails → file fails, trace records error, **no** MarkItDown fallback (per ADR 0001).
