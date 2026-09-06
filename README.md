# 课题组知识助手（wenmai-assistant）

AI 方向研究生独立完成的课题组知识助手：把论文、实验记录、项目文档与组会纪要等资料入库，混合检索，回答必须带出处；库里没有的事实就拒答。

技术对齐 `fjgdAgent/RAG系统.md` 的模块边界（五块存储、Factory、Trace），语料与中文检索自写。

快速讲清仓库结构时，先看 `docs/architecture-legibility.md`；正式术语以 `CONTEXT.md` 为准，架构决策看 `docs/adr/`。

## 本周范围

- 文本 + 图转文 + 本地 `bge-m3` Dense + jieba BM25 + **RRF（已默认）** + Cross-Encoder 精排 + 研究主题过滤（#16、#19 ✅）
- 入库质量门（60/80 三档）+ 灰区复判 + 审阅状态 / 待审（#34–37 ✅）；生成前相邻块扩展（#38 ✅）
- 生成：本地 Gemma MLX（`mlx_vlm.server` :8120），回答带引用，无依据则拒答
- 提问处理：术语归一 + Multi-Query（#42、#43 ✅；失败回退原问）
- 服务与监测：FastAPI + Jinja2；**检索工作台**（`/`）与**运维看板**（`/ops`，含待审与 P50/P95）；MCP `ask.answer`、`collections.*`、`documents.*`、`reviews.*`、`images.*`（统一 envelope）
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
uv sync --extra dev --frozen
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

## 两条入库路径

### 公开 demo 复现

仓库内只提供**可公开复现**的抓取清单与评测尺子，不直接提交正文产物：

```bash
PYTHONPATH=src python scripts/fetch_corpus.py
PYTHONPATH=src python scripts/ingest_corpus_manifest.py
```

这一路径对应 `data/corpus/manifest.yaml` 与 `data/eval/golden.jsonl`，适合演示公开科研资料上的混合检索、拒答与评测。

### 私有科研资料导入

个人文献和组内文档不需要提交进 Git。现有入库接口支持通过 `source_kind` 显式区分两类资料：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "source_path": "/absolute/path/to/paper.pdf",
    "source_kind": "personal_literature"
  }'
```

若目录下是 Zotero 自动导入形成的附件树，也可以直接做个人文献库批量导入：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "source_path": "/absolute/path/to/zotero/storage",
    "source_kind": "personal_literature"
  }'
```

目录导入会递归扫描 `.md` / `.pdf`，沿用现有内容幂等语义跳过未变化文件，并返回 `total`、`attempted`、`ingested`、`skipped`、`failed` 与逐文件摘要。单文件导入仍保持兼容。

对论文类资料，系统会优先保留文件内可得的基础元数据，例如题目、作者、年份；若是 Markdown，也可在 YAML 头里补充：

```yaml
title: Attention Is All You Need
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit
year: 2017
source_kind: personal_literature
```

不依赖 Zotero 数据库本体，也不要求把私有 PDF、组会纪要或组内文档提交到仓库。

Phase B 本机跑批（真实模型，一次只跑一条重链路）：

```bash
PYTHONPATH=src python scripts/ingest_corpus_manifest.py
PYTHONPATH=src python scripts/run_phase_b_batch.py
# 已入库：--skip-ingest
```
