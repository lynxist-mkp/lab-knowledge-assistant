# 福云·文脉助手（wenmai-assistant）

实习场景下的检索助手原型：公开网页 / PDF 入库，混合检索，回答必须带出处。**未接入福云生产库。**

技术对齐 `fjgdAgent/RAG系统.md` 的模块边界（五块存储、Factory、Trace），语料与中文检索自写。

## 本周范围

- 文本 + 图转文 + 本地 `bge-m3` Dense + jieba BM25 + **RRF（已默认）** + Cross-Encoder 精排 + 文化域过滤（#16、#19 ✅）
- 生成：本地 Gemma MLX（`mlx_vlm.server` :8120），回答带引用，无依据则拒答
- 服务与监测：FastAPI + Jinja2 六页 — **总览 / 数据浏览 / Ingestion 管理 / Ingestion 追踪 / Query 追踪已可用**；评估面板待做（#28）
- 评测：黄金集 50 条（#25 ✅）；Hit@5/MRR + 四组消融（#26 ✅，`scripts/run_eval_ablation.py`）；Ragas collections（#27 ✅，缺钥降级）
- 音频（Dolphin 转写）放在文本、图、监测、评测都完成之后（#31）

**进度快照**见 `.scratch/fuyun-wenmai/map.md` 的 Checkpoint 段。

实现规格见 `.scratch/fuyun-wenmai/spec.md`。

## 运行

需要 Python 3.12 与 ModelScope 缓存的模型（生成/图转文 Gemma、Dense `bge-m3`；扫描件 OCR 另需 `paddleocr-env`）。

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

已验证：MPS 可用，`bge-m3` 已在本机 HuggingFace 缓存中。

密钥与 MLX 服务见 `settings.yaml` 与 `docs/adr/`。开发期测试走 fake provider，不加载真实模型。

```bash
uv run pytest

uv run uvicorn wenmai.app:app --factory --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_path": "/absolute/path/to/file.md"}'
```

成功时返回 `document_id`、`chunk_count`、`elapsed_ms`、`trace_id`。Trace 追加写入 `logs/traces.jsonl`。
