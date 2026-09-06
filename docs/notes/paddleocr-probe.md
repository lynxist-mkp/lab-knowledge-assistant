# PaddleOCR-VL / mlx-vlm-server probe (ticket 33)

**Date:** 2026-08-26  
**Goal:** Verify `mlx_vlm.server` can load `mlx-community/PaddleOCR-VL-1.6-5bit` and serve PaddleOCR-VL via `vl_rec_backend="mlx-vlm-server"`.

## Status

**Automated tests:** ✅ 11 个窄缝测试全绿。

**本机 mlx 探路（ModelScope，2026-08-26）：** ✅ 端到端跑通

1. ModelScope 拉 `mlx-community/PaddleOCR-VL-1.6-5bit`
2. `mlx_vlm.server --model <local-path>` 加载成功（`/health` 200）
3. `paddleocr_parse_pdf.py` + `mlx-vlm-server` 解析扫描探针 PDF 成功
4. `POST /ingest` 走 `paddleocr-vl`，chunk 含 `[IMAGE: …]` 占位符，trace `load.method=paddleocr-vl`

隔离 env 还需：`paddlex[ocr]`、`openai>=1.63`（genai-client）、`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`（见 `setup_paddleocr_env.sh`）。

## Quick setup

```bash
cd lab-knowledge-assistant
./scripts/setup_paddleocr_env.sh
./scripts/download_paddleocr_mlx_models.sh   # 预拉 primary + fallback
./scripts/probe_paddleocr_mlx.sh             # 起 server + 跑一页扫描探针 PDF
```

`settings.yaml` 里 `paddleocr.mlx_model` 保持 **ModelScope 仓库 id**（如 `mlx-community/PaddleOCR-VL-1.6-5bit`）。运行时 `MlxVlmServerManager` 会调用 `scripts/resolve_modelscope_model.py` 落到本地缓存，再传给 `mlx_vlm.server --model <path>`。

## ModelScope ids

| 角色 | ModelScope id |
|------|----------------|
| Primary (5bit MLX) | `mlx-community/PaddleOCR-VL-1.6-5bit` |
| Fallback | `PaddlePaddle/PaddleOCR-VL-1.6` |

缓存目录示例：`~/.cache/modelscope/models/mlx-community--PaddleOCR-VL-1.6-5bit/snapshots/master`

## Manual resolve

```bash
source .venvs/paddleocr-env/bin/activate
python scripts/resolve_modelscope_model.py mlx-community/PaddleOCR-VL-1.6-5bit
mlx_vlm.server --port 8111 --model <printed-path>
curl http://127.0.0.1:8111/health
```

## Fallback policy

- Primary ModelScope 拉取或 server 启动失败 → 自动试 `paddleocr.mlx_fallback_model`。
- OCR parse 失败 → 该文件失败、记 trace，**不回退** MarkItDown（ADR 0001）。
