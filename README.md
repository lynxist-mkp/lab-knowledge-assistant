# 福云·文脉助手（wenmai-assistant）

实习场景下的检索助手原型：公开网页 / PDF 入库，混合检索，回答必须带出处。**未接入福云生产库。**

技术对齐 `fjgdAgent/RAG系统.md` 的模块边界（五块存储、Factory、Trace），语料与中文检索自写。

## 本周范围

- 文本 + 图转文 + 本地 `bge-m3` Dense + jieba BM25 + **RRF（已默认）** + Cross-Encoder 精排 + 文化域过滤（#16、#19 ✅）
- 入库质量门（60/80 三档）+ 灰区复判 + 审阅状态 / 待审（#34–37 ✅）；生成前相邻块扩展（#38 ✅）
- 生成：本地 Gemma MLX（`mlx_vlm.server` :8120），回答带引用，无依据则拒答
- 提问处理：术语归一 + Multi-Query（#42、#43 ✅；失败回退原问）
- 服务与监测：FastAPI + Jinja2；**编辑工作台**（`/`）与**运维看板**（`/ops`，含待审与 P50/P95）；MCP `ask_wenmai` + `get_document_summary`
- 评测：黄金集 100 条（#41 ✅）；Hit@5/MRR + 四组消融 + 改写对照（`scripts/run_eval_ablation.py` / `scripts/run_phase_b_batch.py`）；Ragas collections（缺钥降级）
- 演示音频端到端（#31）无限期搁置；Dolphin 适配器（#30）保留

**进度快照**见 `.scratch/fuyun-wenmai/map.md` 的 Checkpoint 段。

实现规格见 `.scratch/fuyun-wenmai/spec.md`。

## 运行

需要 Python 3.12 与 ModelScope 缓存的模型（生成/图转文 Gemma、Dense `bge-m3`；扫描件 OCR 另需 `paddleocr-env`）。

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

已验证：MPS 可用，`bge-m3` 已在本机 HuggingFace 缓存中。

## CI

GitHub Actions 会在对 `main` 的 push 和所有 PR 上执行基础 CI。

本地复现同一套检查：

```bash
uv sync --dev --frozen
uv run ruff check .
uv run pytest
```

密钥与 MLX 服务见 `settings.yaml` 与 `docs/adr/`。开发期测试走 fake provider，不加载真实模型。

**本机生成/图转文默认 Gemma MLX**（`providers.multimodal` = `mlx_gemma`）。首次 ask 或 caption 前建议先起服务（需 Terminal.app + Metal）：

```bash
./scripts/start_gemma_mlx.sh
# 或双击 scripts/start_gemma_mlx.command
```

```bash
uv run pytest

uv run uvicorn wenmai.app:app --factory --host 127.0.0.1 --port 8000
./scripts/stop_server.sh
```

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_path": "/absolute/path/to/file.md"}'
```

成功时返回 `document_id`、`chunk_count`、`elapsed_ms`、`trace_id`。Trace 追加写入 `logs/traces.jsonl`。

Phase B 本机跑批（真实模型，一次只跑一条重链路）：

```bash
PYTHONPATH=src python scripts/ingest_corpus_manifest.py
PYTHONPATH=src python scripts/run_phase_b_batch.py
# 已入库：--skip-ingest
```
